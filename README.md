# Activity Recognition for Autism Detection 

Code for **Activity Recognition for Autism Detection**.

**[Authors](mailto:anish.lakkapragada@gmail.com,peter100@stanford.edu,dpwall@stanford.edu)**: [Anish Lakkapragada](mailto:anish.lakkapragada@gmail.com), [Peter Washington](mailto:peter100@stanford.edu), [Dennis P. Wall](mailto:dpwall@stanford.edu)

<details>
    <summary>Abstract</summary>
     insert final abstract here 
</details>

## Objective 

Today's autism diagnosis is a quite lengthy and inefficient process. Families often have to wait a few years before receiving a diagnosis, and this problem is exacerbated by the fact that the earliest possible intervention is required for best clinical outcomes. One of the biggest indicators of autism is self-stimulatory, or stimming, behaviors such as hand flapping, headbanging, and spinning. In this paper, we demonstrate successful lightweight detection of hand flapping in videos using deep learning and activity recognition. We believe such methods will help create a remote, fast, and accessible autism diagnosis. 

## Data 

In order to collect the videos, we use the [Self-Stimulatory Behavior Dataset (SSBD)](https://rolandgoecke.net/research/datasets/ssbd/) which contains 75 videos of hand flapping, head banging, and spinning behavior. We extract 100 hand flapping and control 2-3 second videos from this dataset and use it 
to train our model. Because most of the videos in SSBD are children, we wanted to see whether a model we built could generalize past age (and hand shape), so we recorded 3 positive and 3 control videos of ourselves. 

Our entire dataset can be found in a google drive folder [here](https://tinyurl.com/47fya6).

## Approach 

In this space, one way to detect hand flapping in videos that has been done before is to use pose estimation images and feed them into time-series based 
Recurrent Neural Networks (RNNs). However because these approaches use heavier feature representations, they demand heavier models; this is not ideal for creating models to deploy into mobile phones, etc. We decide to use a much lighter feature representation and model by using landmark detection to find the
coordinates of the detected hand landmarks as a feature representation over time which is then fed into a Long-Short Term Memory (LSTM) network. 

<img src = "docs/mediapipe_landmarks.png">

The above image shows the 21 landmarks that Mediapipe, a landmark detection framework made by Google, recognizes on the hands. We take the first 90 frames of a video, and at each frame,  take the detected <em> (x, y, z) </em> coordinates of a set amount of landmarks and concatenate them to become a "location frame". Note that the location frame is always 6x bigger than the set amount of landmarks, because there are 3 coordinates for a given landmark and 2 hands where that landmark can be found. If a landmark is not detected, we set all of its <em> (x, y, z) </em> coordinates to 0. This location vector for every frame is fed into an LSTM network that then gives us our prediction. We train models with and without augmentation and compare their results. Our overall approach is shown in the below image. 

<img src = "docs/Approach.png">

It is easy to see that this kind of feature representation and model architecture would easily fit into a model device even unoptimized. Our most heavy model uses <50,000 parameters. 


## Results

## Citation 

## Acknowledgements 