## NFL Trak
Using All-22 footage to track player movement in order to create public access to scheme data, team play data, and player movement data.

![Lead Example](outputs/output_gif_2.gif)

## Overview
This project as it is currently introduces a machine learning pipeline to automatically track football (NFL) players given sideline All-22 footage. 

The system utilizes YOLO-based detection to identify the players in each frame and track them throughout a play using a BoT-SORT algorithm. This is the baseline for a multilevel player tracker that is able to consistently track individual players from the start of a play to the end through collisions, camera movement, and occlusion. 

The current model will serve as a foundation to introduce a competely new way to analyze player performance, team matchups, and individual play data for the next generation of NFL analysts.

## Pipeline
Video Input -> Frame Extraction -> Player Detection via YOLO -> Multi-Object Tracking (Work In Progress) -> Trajectory Analaysis (Work In Progress) -> Tracked Video Output

## Dataset Preperation

Training data was initially manually created (by me) from NFL All-22 Footage using CVAT for annotation. 

Initial Model Training Pipeline:

Video Data -> Frame Extraction from Video -> Initial Data Labeling In CVAT (100% Manual) (6 plays, ~1.5 hours each) -> Model Training in Colab 

Once the model was consistent enough to make it worth it, I was able to use the YOLO annoatations to assist in the training process:

Frame Extraction -> YOLO Labeling -> Annotation Correction in CVAT (10 plays, ~45 min each) -> Model Retraining in Colab

## Model Training

The player detection model was trained using the custom annotated play data.

Training process:

Input: Annotated football video frames
Model: YOLO player detection model
Class: Player
Output: Bounding boxes and confidence scores

Most recent training config: 

- Image Resolution: 768px
- Epochs: 50
- Training Split: 14 Plays
- Test Split: 2 Plays

Additionally, multiple additional data augmentation parameters were used to make the model more robust to camera changes (scale=0.5, translate=0.1, fliplr=0.5, etc.)

## Tracking

Currently, players are tracked across frames using a simple position based BoT-SORT tracking method.

Simple Tracking Pipeline:

1. Detect players in each frame
2. Associate detections between frames utilizing pixel based position

However, this doesn't account for camera movement, lengthened player collisions, or temporary occlusion. The tracking pipeline that's currently under construction will address these issues.

Improved Tracking Pipeline:

1. Detect players in each frame
2. Associate detections between frames utilizing field-based position
3. Analyze movement trajectories to maintain player identities after collisions/occlusion

## Results

The model successfully detects all 22 players throughout a play, but still struggles to keep consistent player identities over time.

![Results Example](outputs/output_gif_1.gif)

## Repository Structure

nfl_trak

|

  |---- models/ # trained yolo model with training results
 
  |---- data/ # since it's NFL data, not freely available, contact me if you're interested in using it and we can probably work something out
 
  |---- training/ # model training python files
 
  |---- tracking/ # player tracking python files
 
  |---- outputs/ # example results
 
  |---- README.md
