####################################################################
#
# wildve.py -- WildVE: Wildlife Video Extractor
#
# Luke Sheneman
# sheneman@uidaho.edu
# 2024-2025
#
# Given a directory of videos, process each video to look for
# wildlife using an ensemble of AI models. Extracts video clips
# which include animals into destination directory.
# Writes summary log and per-process telemetry.
#
# Supports --allframes mode to analyze frame-by-frame and
# report detailed confidences per frame.
#
####################################################################

import os
import sys
import time
import csv
import pathlib
import argparse
from multiprocessing import Process, current_process, freeze_support, Lock, RLock, Manager
import cv2
import math
import nvidia_smi
import logging
import random
import torch
import glob
import numpy as np
import imageio
import open_clip
from ultralytics import YOLO
from tqdm import tqdm	

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy

from PytorchWildlife.models import detection as pw_detection


DEFAULT_INPUT_DIR	 = "inputs"
DEFAULT_OUTPUT_DIR	 = "outputs"
DEFAULT_LOGGING_DIR  	 = "logs"

TIGER_MODEL              = 'best_enlightengan_and_yolov8.pt'
FLORENCE_MODEL           = 'microsoft/Florence-2-large'
CLIP_MODEL               = 'ViT-B/32'

DEFAULT_INTERVAL         = 1.0   # number of seconds between samples
DEFAULT_PADDING		 = 5.0   # number of seconds of video to include before first detection and after last detection in a clip
DEFAULT_REPORT_FILENAME  = "report.csv"
DEFAULT_NPROCS           = 4
DEFAULT_NOBAR		 = False

parser = argparse.ArgumentParser(prog='wildve', description='WildVE: Wildlife Video Extractor - Analyze videos using an ensemble of AI models and extract clips containing wildlife, or analyze frame-by-frame.')

parser.add_argument('input',  metavar='INPUT_DIR',  default=DEFAULT_INPUT_DIR,  help='Path to input directory containing MP4 videos')
parser.add_argument('output', metavar='OUTPUT_DIR', default=DEFAULT_OUTPUT_DIR, help='Path to output directory for clips (if not --allframes) and metadatas')

parser.add_argument('-i', '--interval', type=float, default=DEFAULT_INTERVAL,        help='Number of seconds between AI sampling/detection for clip mode (DEFAULT: '+str(DEFAULT_INTERVAL)+')')
parser.add_argument('-p', '--padding',  type=float, default=DEFAULT_PADDING,         help='Number of seconds of video to pad on front and end of a clip for clip mode (DEFAULT: '+str(DEFAULT_PADDING)+')')
parser.add_argument('-r', '--report',   type=str,   default=DEFAULT_REPORT_FILENAME, help='Name of report metadata CSV file (DEFAULT: '+DEFAULT_REPORT_FILENAME+')')
parser.add_argument('-j', '--jobs',	type=int,   default=DEFAULT_NPROCS,          help='Number of concurrent (parallel) processes (DEFAULT: '+str(DEFAULT_NPROCS)+')')
parser.add_argument('-l', '--logging',  type=str,   default=DEFAULT_LOGGING_DIR,     help='The directory for log files (DEFAULT: '+str(DEFAULT_LOGGING_DIR)+')')

parser.add_argument('-n', '--nobar',    action='store_true',  default=DEFAULT_NOBAR,     help='Turns off the Progress Bar during processing.  (DEFAULT: Use Progress Bar)')
parser.add_argument('--allframes', action='store_true', default=False, help='Analyze every frame and output a detailed CSV report, instead of creating clips.')


group = parser.add_mutually_exclusive_group()
group.add_argument('-g', '--gpu', action='store_true',  default=True, help='Use GPU if available (DEFAULT)')
group.add_argument('-c', '--cpu', action='store_true', default=False, help='Use CPU only')

args = parser.parse_args()

os.environ['YOLO_VERBOSE'] = 'False'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if not os.path.exists(args.input):
	print(f"Error:  Could not find input directory path '{args.input}'", flush=True)
	parser.print_usage()
	sys.exit(-1)

if not os.path.exists(args.output):
	print(f"Could not find output directory path '{args.output}'...Creating Directory!", flush=True)
	os.makedirs(args.output)

if not os.path.exists(args.logging):
	print(f"Could not find logging directory path '{args.logging}'...Creating Directory!", flush=True)
	os.makedirs(args.logging)

if args.cpu:
	device = "cpu"
	usegpu = False
else:
	if torch.cuda.is_available():
		device = "cuda"
		usegpu = True
	else:
		device = "cpu"
		usegpu = False

torch.device(device)

# Global model variables to be initialized per process
megadetector5_model_g = None
megadetector6v9_model_g = None
megadetector6v10_model_g = None
tiger_model_g = None
florence_model_g = None
florence_processor_g = None
clip_model_g = None
clip_processor_g = None

# Global variables for clip mode, managed per process instance
chunk_idx_g = 0
most_recent_written_chunk_g = -1


def load_models(pid):
	global megadetector5_model_g, megadetector6v9_model_g, megadetector6v10_model_g, tiger_model_g, florence_model_g, florence_processor_g, clip_model_g, clip_processor_g

	print(f"PID={pid}: Loading Megadetector 5 model...")
	megadetector5_model_g = pw_detection.MegaDetectorV5(device=device, pretrained=True)

	print(f"PID={pid}: Loading Megadetector 6v9 model...")
	megadetector6v9_model_g = pw_detection.MegaDetectorV6(device=device, version="MDV6-yolov9-e")

	print(f"PID={pid}: Loading Megadetector 6v10 model...")
	megadetector6v10_model_g = pw_detection.MegaDetectorV6(device=device, version="MDV6-yolov10-e")

	print(f"PID={pid}: Loading EnlightenGAN Tiger model...")
	tiger_model_g = YOLO(TIGER_MODEL)

	print(f"PID={pid}: Loading Florence-2 model...")
	florence_model_g = AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL, trust_remote_code=True).eval()
	florence_processor_g = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)

	print(f"PID={pid}: Loading CLIP model...")
	clip_model_g, _, clip_processor_g = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)

	print(f"PID={pid}: All models loaded.")

	print(f"PID={pid}: Deploying applicable models to device: {device}")
	# PytorchWildlife models are already on device from init
	tiger_model_g.to(device)
	florence_model_g.to(device)
	# clip_model is already on device from clip.load()

	print(f"PID={pid}: Models deployed on device: {device}")

	# Return tuple for --allframes mode for easier passing
	return megadetector5_model_g, megadetector6v9_model_g, megadetector6v10_model_g, tiger_model_g, florence_model_g, florence_processor_g, clip_model_g, clip_processor_g


def parse_megadetector_detections(detections):
	megadetector_classes = []
	megadetector_confidences = []
	megadetector_has_animal = False
	megadetector_max_confidence = 0.0

	if detections is not None and hasattr(detections, 'confidence') and detections.confidence is not None:
		if len(detections.confidence) > 0:
			megadetector_classes = detections.class_id.tolist()
			megadetector_confidences = detections.confidence.tolist()
			megadetector_has_animal = any(cls == 0 for cls in megadetector_classes) # MD class 0 is often 'animal'
			megadetector_max_confidence = max(
				(conf for cls, conf in zip(megadetector_classes, megadetector_confidences) if cls == 0),
				default=0.0
				)
	return (
		megadetector_classes,
		megadetector_confidences,
		megadetector_has_animal,
		megadetector_max_confidence
	)


def report_clip_mode(pid, report_list): # Renamed from report()
	filename, clip_path, fps, start_frame, end_frame, confidences = report_list

	min_conf = min(confidences) if confidences else 0
	max_conf = max(confidences) if confidences else 0
	mean_conf = sum(confidences) / len(confidences) if confidences else 0

	s = f'"{filename}", "{clip_path}", {start_frame}, {start_frame/fps:.02f}, {end_frame}, {end_frame/fps:.02f}, {end_frame-start_frame}, {(end_frame-start_frame)/fps:.02f}, {min_conf:.02f}, {max_conf:.02f}, {mean_conf:.02f}\n'

	try:
		# report_lock should be acquired before calling this function
		with open(args.report, "a") as report_file:
			report_file.write(s)
	except:
		print(f"Warning:  Could not open report file {args.report} for writing in report_clip_mode()", flush=True)

def label(img, frame, fps):
	s = f"frame: {frame}, time: {frame/fps:.3f}"
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,0), 6, cv2.LINE_AA) 	
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 3, cv2.LINE_AA) 	
	return img

def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

def reset_screen():
	if os.name != 'nt':
		os.system('reset')

def human_size(bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
	return str(bytes) + units[0] if bytes < 1024 else human_size(bytes>>10, units[1:])

def get_gpu_info():
	try:
		nvidia_smi.nvmlInit()
		deviceCount = nvidia_smi.nvmlDeviceGetCount()
		gpu_info = [deviceCount]
		for i in range(deviceCount):
			handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
			mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
			gpu_info.append((mem_info.total, mem_info.used, mem_info.free))
		nvidia_smi.nvmlShutdown()
		return gpu_info
	except nvidia_smi.NVMLError:
		print("NVIDIA SMI not found or failed to initialize. GPU info not available.", flush=True)
		return [0]


def chunks(filenames, n):
	if n <= 0:
		return []
	k = len(filenames)
	if k == 0:
		return []
	
	chunk_size = k // n
	remainder = k % n
	
	chunk_list = []
	start_index = 0

	for i in range(n):
		end_index = start_index + chunk_size + (1 if i < remainder else 0)
		if end_index > start_index: # Ensure chunk is not empty
			chunk_list.append(filenames[start_index:end_index])
		start_index = end_index
		if start_index >= k and i < n -1 and not any(chunk_list): # if k < n
			# if we have fewer files than processes, some processes get empty chunks
			# this logic ensures we return n chunks, some possibly empty
			for j in range(i + 1, n):
				chunk_list.append([])


	# Ensure we always return n chunks, even if some are empty (e.g., if len(filenames) < n)
	while len(chunk_list) < n and n > 0 :
		chunk_list.append([])
	return chunk_list


def contains_target_label(data):
	target_labels = {"tiger", "tigers", "cat", "wildcat", "animal", "predator", "carnivore"}
	# Florence <OD> task returns a list of dicts, each with 'labels' and 'bboxes'
	# Example: [{'bboxes': [[0.0602, 0.0938, 0.903, 0.8437]], 'labels': ['tiger']}]
	if isinstance(data, dict) and "<OD>" in data: # Old structure
		labels = data.get("<OD>", {}).get("labels", [])
		return any(label in target_labels for label in labels)
	elif isinstance(data, list): # New structure from post_process_generation
		for item in data:
			if "labels" in item:
				if any(label in target_labels for label in item["labels"]):
					return True
	return False


def ensemble_detection(img, md5_model, md6v9_model, md6v10_model, tig_model, flor_model, flor_processor, cl_model, cl_processor):
	# MegaDetector V5
	megadetector5_results    = md5_model.single_image_detection(img)
	megadetector5_detections = megadetector5_results.get('detections', None)
	_, _, megadetector5_has_animal, megadetector5_max_confidence = parse_megadetector_detections(megadetector5_detections)

	# MegaDetector V6v9
	megadetector6v9_results    = md6v9_model.single_image_detection(img)
	megadetector6v9_detections = megadetector6v9_results.get('detections', None)
	_, _, megadetector6v9_has_animal, megadetector6v9_max_confidence = parse_megadetector_detections(megadetector6v9_detections)

	# MegaDetector V6v10
	megadetector6v10_results    = md6v10_model.single_image_detection(img)
	megadetector6v10_detections = megadetector6v10_results.get('detections', None)
	_, _, megadetector6v10_has_animal, megadetector6v10_max_confidence = parse_megadetector_detections(megadetector6v10_detections)

	# Tiger Model (w/EnlightenGAN) Detection
	tiger_results = tig_model(img, verbose=False)
	if isinstance(tiger_results, list):
		tiger_results = tiger_results[0]
	tiger_confidences = [box.conf.item() for box in tiger_results.boxes]
	tiger_has_tiger = any(int(box.cls) == 0 for box in tiger_results.boxes) # Assuming class 0 is tiger
	tiger_max_confidence = max((box.conf.item() for box in tiger_results.boxes if int(box.cls) == 0), default=0.0)


	# Microsoft Florence-2 Detection
	florence_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Florence expects RGB
	florence_parsed_answer = florence_task(florence_image, "<OD>", flor_model, flor_processor)
	florence_detected_target = contains_target_label(florence_parsed_answer)


	# OpenAI CLIP Detection
	img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # CLIP expects RGB
	clip_image_processed = cl_processor(img_pil).unsqueeze(0).to(device)
	text_descriptions = ["a photo of a tiger in a zoo enclosure", "a photo of a zoo enclosure without a tiger", "photo of an animal", "empty photo"]
	tokenizer = open_clip.get_tokenizer('ViT-B-32')
	text_tokens = tokenizer(text_descriptions).to(device)

	with torch.no_grad():
		image_features = cl_model.encode_image(clip_image_processed)
		text_features = cl_model.encode_text(text_tokens)

	image_features /= image_features.norm(dim=-1, keepdim=True)
	text_features  /= text_features.norm(dim=-1, keepdim=True)

	similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()
	
	# Simple logic: highest similarity to "tiger" vs "no tiger"
	# Or, if "tiger" similarity is above a threshold and higher than "empty"
	clip_is_tiger_detection = similarity[0] > similarity[1] and similarity[0] > similarity[3] + 0.05 # Tiger vs no tiger, and tiger vs empty + margin
	clip_tiger_confidence = similarity[0] if clip_is_tiger_detection else 0.0


	results = {
		"megadetector5_detection": megadetector5_has_animal,
		"megadetector5_conf": megadetector5_max_confidence,
		"megadetector6v9_detection": megadetector6v9_has_animal,
		"megadetector6v9_conf": megadetector6v9_max_confidence,
		"megadetector6v10_detection": megadetector6v10_has_animal,
		"megadetector6v10_conf": megadetector6v10_max_confidence,
		"tiger_detection": tiger_has_tiger,
		"tiger_conf": tiger_max_confidence,
		"florence_detection": florence_detected_target, # Boolean
		"clip_detection": clip_is_tiger_detection, # Boolean based on "tiger" prompt
		"clip_conf": clip_tiger_confidence, # Confidence for "tiger"
	}
	return results 


def telemetry_log(log_fd, data):
	filename = log_fd.name
	converted_data = {k: (v.item() if isinstance(v, (np.generic, np.ndarray)) else v) for k, v in data.items()}
	filtered_data = {k: v for k, v in converted_data.items() if isinstance(v, (int, float, str, bool))}
	
	current_chunk_idx = data.get("chunk_idx", -1) # Use passed chunk_idx
	filtered_data = {"chunk_idx": current_chunk_idx, "filename": filename, **filtered_data}

	fieldnames = list(filtered_data.keys())
	if "filename" in fieldnames and fieldnames[1] != "filename": # Ensure filename is second
		fieldnames.insert(1, fieldnames.pop(fieldnames.index("filename")))
	
	# Ensure chunk_idx is first
	if "chunk_idx" in fieldnames and fieldnames[0] != "chunk_idx":
		fieldnames.insert(0, fieldnames.pop(fieldnames.index("chunk_idx")))


	writer = csv.DictWriter(log_fd, fieldnames=fieldnames)
	if log_fd.tell() == 0:
		writer.writeheader()
	writer.writerow(filtered_data)
	log_fd.flush()


def get_video_chunk_clip_mode(invid, interval_sz, pu_lock, log_fd, current_chunk_idx_val):
	# Uses global models initialized by load_models()
	buf = []
	for _ in range(interval_sz):
		success, image = invid.read()
		if success:
			buf.append(image)
		else:
			return None, False # current_chunk_idx_val is not incremented here

	inference_frame = image # Use the last frame of the chunk for inference
	with pu_lock:
		try:
			results = ensemble_detection(inference_frame, 
										 megadetector5_model_g, megadetector6v9_model_g, megadetector6v10_model_g, 
										 tiger_model_g, florence_model_g, florence_processor_g, 
										 clip_model_g, clip_processor_g)
		except Exception as e:
			print(f"Error: Could not run model inference on frame from chunk index: {current_chunk_idx_val}")
			print(f"Exception: {e}", flush=True)
			# Fallback to no detection to avoid crash, or re-raise
			results = {k: False if "detection" in k else 0.0 for k in [
                "megadetector5_detection", "megadetector5_conf", "megadetector6v9_detection", "megadetector6v9_conf", "megadetector6v10_detection", "megadetector6v10_conf", "tiger_detection", "tiger_conf", "florence_detection", "clip_detection", "clip_conf"]}


	res = {
		"chunk_idx": current_chunk_idx_val, # Use passed current_chunk_idx_val
		"buffer": buf,
		"megadetector5_detection": results.get("megadetector5_detection"),
		"megadetector5_conf": results.get("megadetector5_conf", 0),
		"megadetector6v9_detection": results.get("megadetector6v9_detection"),
		"megadetector6v9_conf": results.get("megadetector6v9_conf", 0),
		"megadetector6v10_detection": results.get("megadetector6v10_detection"),
		"megadetector6v10_conf": results.get("megadetector6v10_conf", 0),
		"tiger_detection": results.get("tiger_detection"),
		"tiger_conf": results.get("tiger_conf", 0),
		"florence_detection": results.get("florence_detection"), # bool
		"clip_detection": results.get("clip_detection"), # bool
		"clip_conf": results.get("clip_conf",0), # float
		"overall_detection": False,
		"overall_confidence": 0.0
	}

	total_models = 5
	detections = [
		res["megadetector5_detection"],
		res["megadetector6v9_detection"],
		res["megadetector6v10_detection"],
		res["tiger_detection"],
		res["florence_detection"],
		res["clip_detection"]
	]
	
	confidences_for_detected = [
		res["megadetector5_conf"] if res["megadetector5_detection"] else 0,
		res["megadetector6v9_conf"] if res["megadetector6v9_detection"] else 0,
		res["megadetector6v10_conf"] if res["megadetector6v10_detection"] else 0,
		res["tiger_conf"] if res["tiger_detection"] else 0,
		1.0 if res["florence_detection"] else 0, # Florence is binary
		res["clip_conf"] # clip_conf is already 0 if not detected by its logic
	]
	
	detection_count = sum(d for d in detections if d) # Sum of True values
	res["overall_detection"] = detection_count >= 2 # Arbitrary threshold: at least 2 models agree
	
	if res["overall_detection"]:
		valid_confidences = [c for c,d in zip(confidences_for_detected, detections) if d] # Only conf of models that detected
		if valid_confidences:
			base_confidence = sum(valid_confidences) / len(valid_confidences)
		else: # Should not happen if overall_detection is True and detection_count >= threshold
			base_confidence = 0.0
		
		# Optional: Adjust confidence by how many models agreed
		# adjustment_factor = detection_count / total_models 
		# res["overall_confidence"] = base_confidence * adjustment_factor
		res["overall_confidence"] = base_confidence # Simpler: average confidence of agreeing models
	else:
		res["overall_confidence"] = 0.0

	if log_fd:
		telemetry_log(log_fd, res)
	
	return res, True


def write_clip_clip_mode(clip, frame_chunk, current_most_recent_written_chunk_val):
	if frame_chunk["chunk_idx"] <= current_most_recent_written_chunk_val:
		print(f"***ALERT:  Trying to write the same chunk {frame_chunk['chunk_idx']} twice or out of order!!!  MOST RECENT CHUNK WRITTEN: {current_most_recent_written_chunk_val}")
		return current_most_recent_written_chunk_val # Return unchanged

	for frame in frame_chunk["buffer"]:
		clip.write(frame)
	return frame_chunk["chunk_idx"] # Return new most_recent_written_chunk


def florence_task(image, task_prompt, model, processor):
	# Ensure image is PIL
	if not isinstance(image, Image.Image):
		image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	inputs = processor(text=task_prompt, images=image, return_tensors="pt")
	
	input_ids = inputs["input_ids"].to(device)
	pixel_values = inputs["pixel_values"].to(device)
   
	generated_ids = model.generate(
		input_ids=input_ids,
		pixel_values=pixel_values,
		max_new_tokens=1024,
		early_stopping=False,
		do_sample=False,
		num_beams=3,		
	)
	
	generated_ids = generated_ids.to("cpu") # Move to CPU before decoding
	
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0] # remove verbose=False, deprecated
	
	# The post_process_generation for <OD> expects image_size as (width, height)
	parsed_answer = processor.post_process_generation(
		generated_text,
		task=task_prompt,
		image_size=(image.width, image.height)
	)
	return parsed_answer


# Worker function for clip generation mode
def process_chunk_clip_mode(pid, video_file_chunk, pu_lock, report_lock):
	global megadetector5_model_g, megadetector6v9_model_g, megadetector6v10_model_g, tiger_model_g, florence_model_g, florence_processor_g, clip_model_g, clip_processor_g
	
	# These need to be local to the function call for each process, not global across calls within a process
	# if a process handles multiple videos. They are reset per video.
	# chunk_idx_val = 0 # This will be per video, not global for the process
	# most_recent_written_chunk_val = -1 # This will be per video

	load_models(pid) # Load models into the global_g variables for this process

	print(f"PID={pid}, Processing CHUNK of {len(video_file_chunk)} video(s) in clip mode.")

	# Per-process telemetry log for clip mode
	log_base_filename = os.path.join(args.logging, f"{pid}_clipmode_telemetry.csv")
	# This log will aggregate telemetry for all videos processed by this PID.
	# Consider if separate logs per video are needed or if appending filename to rows is enough.
	# For now, one log per PID.
	log_fd = open(log_base_filename, "w", newline='') 
	print(f"Opened clip mode telemetry log for PID {pid}: {log_base_filename}")


	for vid_idx, filename in enumerate(video_file_chunk):
		# Reset per-video state variables
		chunk_idx_val = 0 
		most_recent_written_chunk_val = -1

		imageio_success = False
		for _ in range(5): # Reduced retries
			try:
				# Using imageio primarily for metadata, cv2 for frame reading
				v_meta = imageio.get_reader(filename, 'ffmpeg')
				nframes_meta = v_meta.count_frames() if hasattr(v_meta, 'count_frames') else None # some formats lack count_frames
				metadata = v_meta.get_meta_data()
				v_meta.close()

				fps = metadata.get('fps', 30.0) # Default fps if not found
				# duration = metadata.get('duration') # Can be derived from nframes/fps if needed
				size = metadata.get('size')
				if nframes_meta is None or size is None: # Fallback if imageio fails
				    cap_temp = cv2.VideoCapture(filename)
				    if not cap_temp.isOpened(): raise RuntimeError("cv2 cannot open for metadata")
				    nframes_meta = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
				    fps = cap_temp.get(cv2.CAP_PROP_FPS)
				    size = (int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT)))
				    cap_temp.release()


				imageio_success = True
				break
			except Exception as e:
				print(f"pid={str(pid).zfill(2)}: WARNING: imageio/cv2 metadata error for {filename}: {e}. Retrying...", flush=True)
				time.sleep(0.5)
		
		if not imageio_success or size is None or fps is None or nframes_meta is None:
			print(f"pid={str(pid).zfill(2)}: FATAL: Could not get metadata for {filename}. Skipping!", flush=True)
			continue
		
		width, height = size
		nframes = nframes_meta


		try:
			invid = cv2.VideoCapture(filename)
			if not invid.isOpened():
				raise RuntimeError("cv2.VideoCapture failed to open file")
		except Exception as e:
			print(f"pid={str(pid).zfill(2)}: Could not read video file: {filename} with OpenCV ({e}), skipping...", flush=True)
			continue

		DETECTION = 500
		SCANNING  = 501
		state = SCANNING

		interval_frames = max(1, int(args.interval * fps)) # Ensure at least 1 frame
		padding_intervals = math.ceil(args.padding * fps / interval_frames) if interval_frames > 0 else 0
		nchunks_total = math.ceil(nframes / interval_frames) if interval_frames > 0 else 0
		
		clip_number = 0
		buffer_chunks = [] # For pre-detection padding
		forward_buf = []   # For post-detection padding checking
		clip_confidences = [] # Confidences for the current clip

		active_clip_writer = None # Holds the VideoWriter object for the current clip
		active_clip_start_frame = 0


		pbar_desc = f"pid={str(pid).zfill(2)} ClipMode {vid_idx+1}/{len(video_file_chunk)}: {os.path.basename(filename)}"
		pbar = tqdm(total=nframes, position=pid, desc=pbar_desc, ncols=100, unit=" frames", leave=False, mininterval=0.5, file=sys.stdout) if not args.nobar else None
		
		processed_frames_count = 0

		# Initial chunk to start the loop
		frame_chunk_data, success = get_video_chunk_clip_mode(invid, interval_frames, pu_lock, log_fd, chunk_idx_val)
		if success:
			chunk_idx_val +=1
			processed_frames_count += len(frame_chunk_data["buffer"])
			if frame_chunk_data["overall_detection"]:
				clip_confidences.append(frame_chunk_data["overall_confidence"])
		if pbar: pbar.update(len(frame_chunk_data["buffer"]) if success and frame_chunk_data else 0)


		while success:
			if chunk_idx_val > nchunks_total + 1 : # +1 to process the last detection fully
				break
			
			current_detection_result = frame_chunk_data["overall_detection"]

			# State transition from SCANNING to DETECTION
			if state == SCANNING and current_detection_result:
				state = DETECTION
				
				fn_base = os.path.basename(filename)
				clip_name = f"{os.path.splitext(fn_base)[0]}_{clip_number:03d}.mp4"
				clip_path = os.path.join(args.output, clip_name)
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
				active_clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
				clip_number += 1
				
				active_clip_start_frame = buffer_chunks[0]["chunk_idx"] * interval_frames if buffer_chunks else frame_chunk_data["chunk_idx"] * interval_frames
				
				for fc_past in buffer_chunks: # Write pre-padding
					most_recent_written_chunk_val = write_clip_clip_mode(active_clip_writer, fc_past, most_recent_written_chunk_val)
				buffer_chunks = []
				most_recent_written_chunk_val = write_clip_clip_mode(active_clip_writer, frame_chunk_data, most_recent_written_chunk_val) # Write current detecting chunk

			# In DETECTION state
			elif state == DETECTION:
				if current_detection_result: # Still detecting
					if buffer_chunks: # Write any buffered non-detecting chunks from forward_buf logic
						for fc_buffered in buffer_chunks:
							most_recent_written_chunk_val = write_clip_clip_mode(active_clip_writer, fc_buffered, most_recent_written_chunk_val)
						buffer_chunks = []
					most_recent_written_chunk_val = write_clip_clip_mode(active_clip_writer, frame_chunk_data, most_recent_written_chunk_val)
				
				else: # Current chunk is NOT detecting, check padding
					forward_buf = [frame_chunk_data] # Start forward buffer with current non-detecting chunk
					found_detection_in_forward_padding = False
					
					for _ in range(padding_intervals): # Read ahead for padding_intervals
						next_frame_chunk_data, next_success = get_video_chunk_clip_mode(invid, interval_frames, pu_lock, log_fd, chunk_idx_val)
						if next_success:
							chunk_idx_val +=1
							processed_frames_count += len(next_frame_chunk_data["buffer"])
							if next_frame_chunk_data["overall_detection"]:
								clip_confidences.append(next_frame_chunk_data["overall_confidence"])
						if pbar: pbar.update(len(next_frame_chunk_data["buffer"]) if next_success and next_frame_chunk_data else 0)

						if not next_success: break 
						forward_buf.append(next_frame_chunk_data)
						if next_frame_chunk_data["overall_detection"]:
							found_detection_in_forward_padding = True
							break # Detection found, stop reading ahead for this padding check
					
					if not found_detection_in_forward_padding: # End of event, write padding and close clip
						state = SCANNING # Transition back to SCANNING
						
						# Write up to padding_intervals from forward_buf
						write_count = 0
						for i in range(min(padding_intervals, len(forward_buf))):
							most_recent_written_chunk_val = write_clip_clip_mode(active_clip_writer, forward_buf[i], most_recent_written_chunk_val)
							write_count +=1
						
						# Remaining forward_buf items become the new buffer_chunks for pre-padding the *next* potential event
						buffer_chunks = forward_buf[write_count:]
						forward_buf = []

						if active_clip_writer:
							active_clip_writer.release()
							clip_end_frame_num = (most_recent_written_chunk_val * interval_frames) + interval_frames
							with report_lock:
								report_clip_mode(pid, [filename, clip_path, fps, active_clip_start_frame, clip_end_frame_num, clip_confidences])
							active_clip_writer = None
							clip_confidences = [] # Reset for next clip
					
					else: # Detection found within forward padding, continue current clip event
						# Write all chunks from forward_buf up to and including the detected one
						last_detection_idx_in_forward = -1
						for i, f_chunk in enumerate(forward_buf):
							if f_chunk["overall_detection"]: last_detection_idx_in_forward = i
						
						for i in range(last_detection_idx_in_forward + 1):
							most_recent_written_chunk_val = write_clip_clip_mode(active_clip_writer, forward_buf[i], most_recent_written_chunk_val)
						
						# Remaining forward_buf items (if any, after the last detection) become pre-padding buffer
						buffer_chunks = forward_buf[last_detection_idx_in_forward + 1:]
						forward_buf = []
						# State remains DETECTION
			
			# In SCANNING state and current chunk is NOT detecting
			elif state == SCANNING and not current_detection_result:
				buffer_chunks.append(frame_chunk_data)
				if len(buffer_chunks) > padding_intervals:
					buffer_chunks.pop(0) # Keep buffer size limited to padding length

			# Get next chunk for the main loop
			if state == SCANNING or (state == DETECTION and current_detection_result): 
				# If we didn't already advance chunk_idx_val in a look-ahead loop
				frame_chunk_data, success = get_video_chunk_clip_mode(invid, interval_frames, pu_lock, log_fd, chunk_idx_val)
				if success:
					chunk_idx_val +=1
					processed_frames_count += len(frame_chunk_data["buffer"])
					if frame_chunk_data["overall_detection"] and state == DETECTION : # Add confidence if part of an ongoing clip
						clip_confidences.append(frame_chunk_data["overall_confidence"])
					elif frame_chunk_data["overall_detection"] and state == SCANNING : # Potential start of new clip
						clip_confidences = [frame_chunk_data["overall_confidence"]]


				if pbar: pbar.update(len(frame_chunk_data["buffer"]) if success and frame_chunk_data else 0)
			elif state == DETECTION and not current_detection_result:
				# This case means we handled look-ahead. frame_chunk_data for next main loop iteration
				# should be the first item from buffer_chunks (if any) or a new read.
				if buffer_chunks:
					frame_chunk_data = buffer_chunks.pop(0)
					success = True # Assumed, as it was successfully read before
				else: # Buffer empty, need to read new chunk
					frame_chunk_data, success = get_video_chunk_clip_mode(invid, interval_frames, pu_lock, log_fd, chunk_idx_val)
					if success:
						chunk_idx_val +=1
						processed_frames_count += len(frame_chunk_data["buffer"])
						if frame_chunk_data["overall_detection"] and state == DETECTION : # Add confidence if part of an ongoing clip
							clip_confidences.append(frame_chunk_data["overall_confidence"])
						elif frame_chunk_data["overall_detection"] and state == SCANNING : # Potential start of new clip
							clip_confidences = [frame_chunk_data["overall_confidence"]]

					if pbar: pbar.update(len(frame_chunk_data["buffer"]) if success and frame_chunk_data else 0)


		# End of video processing loop
		if active_clip_writer: # If a clip was active when video ended
			# Write any remaining padding from buffer_chunks (if logic allows) or just close
			for fc_final_pad in buffer_chunks: # Unlikely to have much here if padding logic is correct
				 most_recent_written_chunk_val = write_clip_clip_mode(active_clip_writer, fc_final_pad, most_recent_written_chunk_val)

			active_clip_writer.release()
			clip_end_frame_num = min(nframes, (most_recent_written_chunk_val * interval_frames) + interval_frames)
			with report_lock:
				report_clip_mode(pid, [filename, clip_path, fps, active_clip_start_frame, clip_end_frame_num, clip_confidences])

		if pbar: 
			if processed_frames_count < nframes: # If loop exited early
				pbar.update(nframes - processed_frames_count) # Ensure pbar completes
			pbar.close()
		invid.release()
		print(f"PID={pid}: Finished processing {filename} in clip mode.")

	log_fd.close()
	print(f"PID={pid}: Closed clip mode telemetry log: {log_base_filename}")


# Worker function for --allframes mode
def process_video_allframes_mode(pid, filename, models_tuple, pu_lock, report_lock, allframes_report_path):
    megadetector5_model, megadetector6v9_model, megadetector6v10_model, tiger_model, florence_model, florence_processor, clip_model, clip_processor = models_tuple

    try:
        invid = cv2.VideoCapture(filename)
        if not invid.isOpened():
            print(f"pid={pid}: Could not open video file for allframes: {filename}, skipping...", flush=True)
            return

        fps = invid.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0 # Default if fps is 0
        total_frames = int(invid.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar_desc = f"pid={str(pid).zfill(2)} AllFrames: {os.path.basename(filename)}"
        pbar = tqdm(total=total_frames, position=pid, desc=pbar_desc, ncols=100, unit=" frames", leave=False, mininterval=0.2, file=sys.stdout) if not args.nobar else None

        frame_num = 0
        while True:
            success, frame = invid.read()
            if not success:
                break

            timestamp = frame_num / fps if fps > 0 else 0

            with pu_lock: # Protects model inference calls
                try:
                    results = ensemble_detection(frame,
                                                 megadetector5_model, megadetector6v9_model, megadetector6v10_model,
                                                 tiger_model, florence_model, florence_processor,
                                                 clip_model, clip_processor)
                except Exception as e:
                    print(f"Error during ensemble_detection for frame {frame_num} of {filename}: {e}", flush=True)
                    if pbar: pbar.update(1)
                    frame_num += 1
                    continue
            
            md5_conf = results.get("megadetector5_conf", 0.0) if results.get("megadetector5_detection") else 0.0
            md6v9_conf = results.get("megadetector6v9_conf", 0.0) if results.get("megadetector6v9_detection") else 0.0
            md6v10_conf = results.get("megadetector6v10_conf", 0.0) if results.get("megadetector6v10_detection") else 0.0
            florence_conf = 1.0 if results.get("florence_detection") else 0.0
            clip_tiger_conf = results.get("clip_conf", 0.0) # Already specific to tiger
            tiger_model_conf = results.get("tiger_conf", 0.0) # Already specific to tiger

            all_confidences = [md5_conf, md6v9_conf, md6v10_conf, florence_conf, clip_tiger_conf, tiger_model_conf]
            mean_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

            report_data = {
                "ORIGINAL": os.path.basename(filename), "FRAME": frame_num, "TIMESTAMP": f"{timestamp:.3f}",
                "megadetectorv5_conf": f"{md5_conf:.4f}", "megadetectorv6v9_conf": f"{md6v9_conf:.4f}",
		"megadetectorv6v10_conf": f"{md6v10_conf:.4f}", "florence_conf": f"{florence_conf:.4f}", 
		"clip_conf": f"{clip_tiger_conf:.4f}", "tiger_model_conf": f"{tiger_model_conf:.4f}", 
		"mean_confidence": f"{mean_conf:.4f}"
            }

            with report_lock: # Protects CSV writing
                with open(allframes_report_path, "a", newline='') as rf:
                    writer = csv.DictWriter(rf, fieldnames=report_data.keys())
                    writer.writerow(report_data)
            
            if pbar: pbar.update(1)
            frame_num += 1

        invid.release()
        if pbar: pbar.close()
        print(f"PID={pid}: Finished processing {filename} in --allframes mode.")

    except Exception as e:
        print(f"pid={pid}: Error processing video {filename} in allframes mode: {e}", flush=True)
        if pbar: pbar.close() # Ensure pbar is closed on error


# Wrapper for --allframes mode to load models and iterate videos in a chunk
def process_chunk_allframes_wrapper(pid, video_files_chunk, pu_lock, report_lock, allframes_report_path):
    print(f"PID={pid}: Initializing models for --allframes mode...")
    models_tuple = load_models(pid) # Returns the tuple of loaded models

    print(f"PID={pid}, Processing CHUNK of {len(video_files_chunk)} video(s) in --allframes mode.")
    for filename in video_files_chunk:
        process_video_allframes_mode(pid, filename, models_tuple, pu_lock, report_lock, allframes_report_path)
    
    print(f"PID={pid}: Finished processing its chunk in --allframes mode.")


def main():
	all_start_time = time.time()

	if usegpu:
		gpu_info = get_gpu_info()
		if gpu_info and gpu_info[0] > 0:
			print(f"Detected {gpu_info[0]} CUDA GPUs")
			for g_idx in range(gpu_info[0]):
				mem_total, mem_used, mem_free = gpu_info[g_idx+1]
				print(f"GPU:{g_idx}, Memory : ({100*mem_free/mem_total:.2f}% free): {human_size(mem_total)}(total), {human_size(mem_free)} (free), {human_size(mem_used)} (used)")
		elif not gpu_info: # Failed to init SMI
			print("Could not get GPU info via nvidia-smi. Proceeding with configured device.", flush=True)
		else: # No GPUs detected
			print("No CUDA GPUs detected by nvidia-smi. Will use CPU if 'cpu' was not specified and CUDA was thought available.", flush=True)


	freeze_support()

	report_file_path = os.path.join(args.output, args.report) # Place report in output dir

	if args.allframes:
		print("*********************************************")
		print("          RUNNING IN ALLFRAMES MODE          ")
		print(" Clips will NOT be generated. Frame-by-frame analysis report.")
		print(f" Report will be saved to: {report_file_path}")
		print("*********************************************\n")
		if (args.interval != DEFAULT_INTERVAL or args.padding != DEFAULT_PADDING):
			print("Warning: --interval and --padding arguments are ignored when --allframes is used.", flush=True)

		try:
			with open(report_file_path, "w", newline='') as rf:
				fieldnames = ["ORIGINAL", "FRAME", "TIMESTAMP", 
							  "megadetectorv5_conf", "megadetectorv6_conf", 
							  "florence_conf", "clip_conf", "tiger_model_conf", 
							  "mean_confidence"]
				writer = csv.DictWriter(rf, fieldnames=fieldnames)
				writer.writeheader()
		except Exception as e:
			print(f"Error: Could not create allframes report file {report_file_path}: {e}", flush=True)
			sys.exit(-1)
	else: # Clip generation mode
		try:
			with open(report_file_path, "w", newline='') as rf: # Use newline='' for csv
				rf.write("ORIGINAL,CLIP,START_FRAME,START_TIME,END_FRAME,END_TIME,NUM_FRAMES,DURATION,MIN_CONF,MAX_CONF,MEAN_CONF\n")
		except:
			print(f"Error: Could not open report file {report_file_path} in main() for clip mode.", flush=True)
			sys.exit(-1)


	print('''
	 __        __ _  _      _ __     __ _____
	 \ \      / /(_)| |  __| |\ \   / /| ____|
	  \ \ /\ / / | || | / _` | \ \ / / |  _|
	   \ V  V /  | || || (_| |  \ V /  | |___
	    \_/\_/   |_||_| \__,_|   \_/   |_____|
	     W i l d l i f e   V i d e o   E x t r a c t o r
	''', flush=True)

	print("            BEGINNING PROCESSING          ")
	print("*********************************************")
	print("           INPUT_DIR: ", args.input)
	print("          OUTPUT_DIR: ", args.output)
	if not args.allframes:
		print("   SAMPLING INTERVAL: ", args.interval, "seconds")
		print("    PADDING DURATION: ", args.padding, "seconds")
	print("    CONCURRENT PROCS: ", args.jobs)
	print("DISABLE PROGRESS BAR: ", args.nobar)
	print("             USE GPU: ", usegpu, f" (Device: {device})")
	print("         REPORT FILE: ", report_file_path)
	print("*********************************************\n\n", flush=True)

	path = os.path.join(args.input, "*.mp4") # Consider other video types if needed
	files = glob.glob(path)
	if not files:
		print(f"No .mp4 files found in {args.input}. Exiting.", flush=True)
		sys.exit(0)

	random.shuffle(files)
	video_file_chunks = chunks(files, args.jobs)

	manager = Manager()
	pu_lock = manager.Lock()     # For synchronizing model inference if necessary
	report_lock = manager.Lock() # For synchronizing writes to the report CSV

	if usegpu:
		torch.cuda.empty_cache()

	processes = []
	target_function = process_chunk_allframes_wrapper if args.allframes else process_chunk_clip_mode
	target_args_suffix = (pu_lock, report_lock, report_file_path) if args.allframes else (pu_lock, report_lock)


	for pid, chunk_of_videos in enumerate(video_file_chunks):
		if not chunk_of_videos: # Skip starting a process for an empty chunk
			print(f"PID={pid} has no videos assigned, skipping process creation.")
			continue
		
		current_target_args = (pid, chunk_of_videos) + target_args_suffix
		p = Process(target=target_function, args=current_target_args)
		processes.append(p)
		p.start()
	
	active_processes = True
	while active_processes:
		active_processes = False
		for p in processes:
			if p.is_alive():
				active_processes = True
			if p.exitcode is not None and p.exitcode != 0:
				print(f"Terminating due to failure in process {p.pid} (exit code {p.exitcode})")
				for proc_to_kill in processes:
					if proc_to_kill.is_alive():
						proc_to_kill.terminate()
				time.sleep(2) # Give processes time to terminate
				# clear_screen() # Optional: clear screen
				# reset_screen() # Optional: reset screen

				print("\n")
				print("*****************************************************************************")
				print("A PROCESS FAILED UNEXPECTEDLY:")
				print("This might be due to insufficient system resources (e.g., GPU RAM).")
				print("Consider reducing the number of concurrent jobs (i.e., --jobs <n>) and try again.")
				print("Check logs in the logging directory for more details.")
				print("*****************************************************************************")
				print("\n\n")
				sys.exit(p.exitcode) # Exit main with the error code
		if active_processes:
			time.sleep(0.5)

	# Final join for all processes that might have finished normally
	for p in processes:
		p.join()


	print(f"\nTotal time to process {len(files)} videos: {time.time()-all_start_time:.02f} seconds")
	print(f"Report file saved to {report_file_path}")
	print("\nDONE\n")

if __name__ == '__main__':
	# Recommended for PyTorch multiprocessing, especially with CUDA
	# 'spawn' is generally safer than 'fork' on systems that support it (macOS, Windows).
	# Linux defaults to 'fork', which can sometimes cause issues with CUDA.
	# Setting it explicitly can improve cross-platform consistency.
	if sys.platform != 'win32': # 'spawn' is default on Windows
		try:
			torch.multiprocessing.set_start_method('spawn', force=True) 
		except RuntimeError as e:
			print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Using default.", flush=True)
	main()
