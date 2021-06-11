Self-Stimulatory Behaviours in the Wild for Autism Diagnosis - Dataset (SSBD)
-----------------------------------------------------------------------------

This package contains

1. ssbd.m - matlab example code for classifying stimming actions using SSBD dataset
2. url-list.pdf - list of URLs with annotations of behaviour instant. This is just for a quick preview. More details on the annotation for every video is found in the Annotations folder
3. ssbd-paper-PID2968273.pdf - Original paper

4. baseline-results.xls - Classification performance on SSBD dataset.

5. Annotations folder containing XML based annotations for all the videos. The tag descriptions are given below

url - location of video 

keyword - title as provided by the content owner. This can be used for searching in YouTube

height, width, frames - specifies the resolution and number of frames in the video

persons - number of people in the video. The video may contain multiple people and children

duration - time duration of the video in seconds

conversation - does the video contain conversation between two people, mostly, between a parent and a child

behaviours - Number of behaviours observed in the video 
time - instant in the video when the behaviour occurs in the starttime::endtime format. The time is specified in seconds

bodypart - the dominant body part involved in the behaviour

category - stimming behaviour category - There is a possibility of behaviours from multiple categories in a single video

intensity - low and high indicating the severity of the behaviour

modality - predominant mode in which the behaviour is observed, say video, speech, etc. This is for future use
 


Installation Instructions

1. Install VLFeat library - http://www.vlfeat.org/
2. Execute ssbd.m in Matlab. The code is tested in Linux platform. The code is self-contained and it downloads the SSBD dataset (stip files of videos) and runs the program. Please see the comments in the code for more information
3. Please use the results given in the baseline-results.xls (available in this package) for benchmarking. The results are slightly different from the one published in paper. This is due to incorrect usage of a video earlier and also due to initialization of random seed for visual words construction. The baseline-results.xls correspond to results obtained in Linux platform. Sorry for the confusion.


Please contact Shyam Sundar Rajagopalan (Shyam.Rajagopalan@canberra.edu.au) for any assistance and suggestions.






