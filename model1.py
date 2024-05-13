import cv2
import time
import datetime
import asone
from asone import ASOne
import sys
import argparse

import socket
import pickle
import os
import ftplib
import json
import paho.mqtt.client as mqtt

from loguru import logger
from datetime import datetime, timezone, timedelta
import enum

# Calculate the center coordinates of bounding box
def bb_center(bbox, i):
    return (float((bbox[i][2] + bbox[i][0]) / 2), float((bbox[i][3] + bbox[i][1]) / 2))

def bb_info(bbox, i):###
    width = float(bbox[i][2] - bbox[i][0])
    height = float(bbox[i][3] - bbox[i][1])
    return (float((bbox[i][2] + bbox[i][0]) / 2), float((bbox[i][3] + bbox[i][1]) / 2), height ,width )

# Calculate the square distance between two center points
def sq_dist(c1, c2):
    return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2

# Function to send recorded evidence video to backend via FTP
def send_video(video_path):

    filename = os.path.basename(video_path)

	# Define FTP connection parameters
    FTP_HOST = '192.168.0.207'
    FTP_USER = 'intellisys'
    FTP_PASS = 'intellisys'
    FTP_DIR = 'Demo/Recorded_Videos'

	# Connect to the FTP server
    ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
    ftp.cwd(FTP_DIR)

	# Upload the video to Backend
    try:
        with open(video_path, 'rb') as file:
            retcode = ftp.storbinary(f'STOR {filename}', file, blocksize=1024*1024)

        if retcode.startswith('226'):
            logger.success(f'Video upload successful: {filename}')
            os.remove(video_path)
        else:
            logger.error(f'Video upload failed: {filename}')
    
    except:
        logger.error(f'Video upload error: {filename}')
    
    finally:
        ftp.quit()

# Function to send the detected object details to the backend
def send_message(new_obj_detect, time_iso, path, new_obj_id, Box_info,color_at_obj_pos): ###
	mqtt_broker = '192.168.0.207'
	mqtt_port = 1883
	topic='CS2/GarbageDetection'
   
	r, g, b = color_at_obj_pos[2], color_at_obj_pos[1], color_at_obj_pos[0]
	color_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)

    # create dictionary
	message_dict = {
        "Id":path,
        "VideoSource":
            {"Name": "camera1",
             "Type":"Network",
             "Url":None,
            },
        "Date": time_iso,
        "Image": "Lab camera",
        "Video": "Lab camera",
        "Category":"VA",
        "Metadata":{
            "Count":1,
             "Detections":
                {"ObjectId":new_obj_id,
                  "ObjectType":"Object",
                  "ObjectSubType":new_obj_detect,
                  "ROIName":"HDB",
                  "Color":color_hex,
                  "Size":"Unkown",
                  "Box":Box_info
                }            
		}
    }
    # dictionary to JSON
	message_json = json.dumps(message_dict)

	client=mqtt.Client()
	try:
		client.connect(mqtt_broker, mqtt_port, 60)
		client.loop_start()  
		client.publish(topic, message_json, qos=0)  #qos could be alter to 1 in the future
		client.loop_stop() 
		
	except Exception as e:
		logger.error(f'Error sending message: "{message_json}", Error: {str(e)}')
	finally:
		client.disconnect()  
		logger.info(f'Sent object details: "{message_json}"')



# Function to record video using saved frames in frame_buffer
def record_video(frame_buffer,new_obj_detect):
	
	#now = datetime.now()
	current_time = datetime.now(tz=timezone(timedelta(hours=8)))   
	time_iso = current_time.isoformat()#ISO 8601 string
	timestamp=current_time.strftime('%Y-%m-%d_%H%M%S')
	#timestamp = f'{str(now.date())}_{str(now.hour)}{str(now.minute)}{str(now.second)}'
	path = f'{timestamp}.mp4'
	logger.info(f'Recording the evidence video {path}')
	output = cv2.VideoWriter(path, fourcc, 12,
							(width, height))
	for frame in frame_buffer:
		output.write(frame)
		k = cv2.waitKey(24)
		if k == ord("q"):
			break
	output.release()
	send_video(path)
	send_message(new_obj_detect, time_iso,path, new_obj_id, Box_info,color_at_obj_pos)
	

if __name__ == '__main__':

	logger.info(f'Starting the Garbage Dumping detection System by CS2')

	class_names = ['Bicycle', 'Chair', 'Box', 'Table', 'Plastic bag', 'Flowerpot', 'Luggage and bags', 'Umbrella', 'Shopping trolley', 'Person']
	stream_url = 'rtsp://cs2projs:cs2projs@192.168.0.166/stream2'
	cap = cv2.VideoCapture(stream_url)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	fps = cap.get(cv2.CAP_PROP_FPS)
	path = ""
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')

	mysocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	mysocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
	server_ip = '192.168.0.207'
	server_port = 8500

	use_cuda = False
	runstatus = False

	while(True):
		starting_status = False
		recording_status = False
		buffer_size = 10
		frame_buffer = []
		start_time = time.time()

		current_obj_ids = []
		new_obj_detect = ''
		new_obj_id = ''
		person_id = ''

		is_stopping = False
		upload_status = False
		video_discard = False
		
		detect = ASOne(tracker=asone.NORFAIR, detector=asone.YOLOV8N_PYTORCH, weights='yolov8_hdb.pt', use_cuda = use_cuda )
		track = detect.track_video(stream_url, class_names=class_names)
		for bbox_details, frame_details in track:
			bbox_xyxy, ids, scores, class_ids = bbox_details
			frame, frame_num, fps = frame_details
			class_list = [class_names[int(i)] for i in class_ids]

			cv2.imshow('Live Video', frame)

			if(recording_status == False):
				has_person = class_list.count('Person') > 0
				for i, obj in enumerate(class_list):
					is_new_object = obj != 'Person' and ids[i] not in current_obj_ids
				
					if has_person and is_new_object:
						new_obj_detect = obj
						new_obj_id = ids[i]
						current_obj_ids.append(ids[i])
						obj_pos = bb_center(bbox_xyxy, i)
						color_at_obj_pos = frame[int(obj_pos[1]), int(obj_pos[0])]
						
						
						Box_info=bb_info(bbox_xyxy, i)   ###
						person_dist = -1

						for j, person in enumerate(class_list):	
							if person == 'Person':
								person_pos = bb_center(bbox_xyxy, j)
								new_person_dist = sq_dist(obj_pos, person_pos)
								
								if new_person_dist < person_dist or person_dist == -1:
									person_dist = new_person_dist
									person_id = ids[j]

						logger.info(f'Person id: {person_id}, {new_obj_detect} id: {new_obj_id} detected.')
						recording_status = True
						break
					#if (recording_status==True):
						#frame_buffer.append(frame)
						#break

			# Saving frsmes
			if (recording_status == True):
				frame_buffer.append(frame)

			# Send frames for Live streaming
			ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 8])
			x_as_bytes = pickle.dumps(buffer)
			mysocket.sendto((x_as_bytes), (server_ip, server_port))

			if (recording_status == True):
				if (person_id not in ids) and (is_stopping == False):
					logger.debug(f'Person id: {person_id} left, waiting for 5 seconds.')
					time_to_stop = time.time() + 7
					is_stopping = True

				if (is_stopping == True):
					if person_id in ids:
						logger.debug(f'Person id: {person_id} redetected, resuming recording.')
						is_stopping = False

					elif time.time() > time_to_stop:
						if new_obj_id in ids:
							upload_status = True
						else:
							logger.debug(f'Object id: {new_obj_id} not found, Video Discarding.')
							video_discard= True
						is_stopping = False


			#Recording,Video Saving and Uploading
			if(upload_status == True and recording_status == True):
				record_video(frame_buffer,new_obj_detect)
				recording_status = False
				upload_status = False
				frame_buffer = []
			
			if(video_discard == True):
				frame_buffer = []
				recording_status = False
				video_discard = False

			if (class_list.count('Person')==0):
				for i, obj in enumerate(class_list):
					if obj != 'Person' and ids[i] not in current_obj_ids:
						current_obj_ids.append(ids[i])
							
		cap.release()
		cv2.destroyAllWindows()
