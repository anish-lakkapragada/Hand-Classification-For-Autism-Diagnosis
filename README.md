
# Code for [Classification of Hand Movement for Autism Detection](https://arxiv.org/abs/2108.07917)


<h1> Recognition </h1> 

Paper is published at [JMIR Biomedical Engineering 2022](https://biomedeng.jmir.org/) and presented at the [DataBricks Summit 2022](https://databricks.com/dataaisummit/north-america-2022). 

<details>
   <summary>Abstract</summary>
   <p> <em> A formal autism diagnosis can be an inefficient and lengthy process. Families may wait months or longer before receiving a diagnosis for their child despite evidence that earlier intervention leads to better treatment outcomes. Digital technologies which detect the presence of behaviors related to autism can scale access to pediatric diagnoses. This work aims to demonstrate the feasibility of deep learning technologies for detecting hand flapping from unstructured home videos as a first step towards validating whether models and digital technologies can be leveraged to aid with autism diagnoses. We used the Self-Stimulatory Behavior Dataset (SSBD), which contains 75 videos of hand flapping, head banging, and spinning exhibited by children. From all the hand flapping videos, we extracted 100 positive and control videos of hand flapping, each between 2 to 5 seconds in duration. Utilizing both landmark-driven-approaches and MobileNet V2's pretrained convolutional layers, our highest performing model achieved a testing F1 score of 84% (90% precision and 80% recall) when evaluating with 5-fold cross validation 100 times. This work provides the first step towards developing precise deep learning methods for activity detection of autism-related behaviors. </em> </p>
</details>

<h1> Summary </h1> 

<h2> Objective </h2> 

Detect autism-related behavior from about a hundred crowdsourced videos of the <a href="https://rolandgoecke.net/research/datasets/ssbd/">Self-Stimulatory Behavior Dataset (SSBD)</a> using machine learning models. 

<h2> Evaluation Method </h2> 

All our results are the average of a 100 K-fold (<em>k</em> = 5) cross validation evaluations on 100 different arrangements of the data (essentially training and evaluation on 500 different datasets). We track the training and testing precision, recall, and F1 score. 

<h2> Results </h2> 

   <details>
      <summary> Method Explanation </summary>
      <p> If interested, we detail all our approaches. 
      For each frame in a given video, we feed it through a feature extractor. The goal of the feature extractor is to map the high dimensional image into a compact vector presentation for each timestep in the LSTM network. <br> <br> 
      We try using MediaPipe's hand landmark detection model to get the numerical coordinates of the hands in each frame (which are then fed to the LSTM) as the feature extractor. MediaPipe by default detects the <em> (x, y, z) </em> coordinates of 21 landmarks on the hand. We explore using subsets (e.g. using six of the landmarks or only one) provided by MediaPipe in the vector representation fed into the LSTM. We also try a "mean landmark" approach where we insert the mean of all the detected landmarks' coordinates. <br> <br> Another approach we tried was to take MobileNet V2's convolutional layers, pretrained on ImageNet, and use those as our feature extractor to convert each frame into a compact vector. </p>
   </details>
   <table>
      <thead>
         <tr>
            <th>Approach</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>AUROC</th>
         </tr>
      </thead>
      <tbody>
         <tr>
            <td> All Landmarks</div></td>
            <td> 68.0 ± 2.66 </td> 
		   <td> 70.3 ± 3.6 </td> 
		   <td> 65.34 ± 5.0 </td> 
		   <td> 66.6 ± 3.35 </td> 
		   <td> 0.748 ± 0.26 </td>     
    </tr>
         <tr>
            <td>  Mean Landmark </div></td>
            <td> 65.5 ± 4.5 </td> 
		   <td> 66.7 ±  7.4 </td> 
		   <td> 66.9 ± 9.6  </td> 
		   <td> 64.2 ± 6.8 </td> 
		   <td> 0.73 ± 0.04 </td>     
         </tr>
         <tr>
	            <td>  One Landmark </div></td>
	            <td> 65.8 ± 4.3 </td> 
			   <td> 66.5 ±  7.5 </td> 
			   <td> 68.0 ± 6.7  </td> 
			   <td> 64.9 ± 6.5 </td> 
			   <td> 0.751 ± 0.03 </td>
         </tr>
         <tr>
	            <td>  Six Landmarks </div></td>
	            <td> 69.55 ± 2.7 </td> 
			   <td> 71.7 ±  3.5 </td> 
			   <td> 67.5 ± 5.5  </td> 
			   <td> 68.3 ± 3.6 </td> 
			   <td> 0.76 ± 0.027 </td>
         </tr>
         <tr>
         	   <td> <strong> MobileNet V2 </strong> </td>
			   <td> <strong> 85.0 ± 3.14 </strong> </td> 
			   <td> <strong> 89.6 ±  4.3 </strong> </td> 
			   <td> <strong> 80.4 ± 6.0 </strong>  </td> 
			   <td> <strong> 84.0 ± 3.7 </strong> </td> 
			   <td> <strong> 0.85 ± 0.03 </strong> </td>
		</tr>
      </tbody>
   </table>

<h2> Code </h2> 

<p>
   We maintain all code here. A demo of our stored and trained MobileNet model is in the `demo` folder. We store our notebooks that we used to evaluate our models in `main-notebooks` and their associated serialized results in `main-results`. Our figures used in the paper are in `plots`. 

   `misc/` contains other methods we tried (e.g. our code for the ensemble classifier) which may or may not have been in the paper. `misc/original-landmark-notebooks` stores the original notebooks for the landmark code. 
</p>

## Citations

Please cite the following:
```
@article{lakkapragada2021classification,
  title={Classification of Abnormal Hand Movement for Aiding in Autism Detection: Machine Learning Study},
  author={Lakkapragada, Anish and Kline, Aaron and Mutlu, Onur Cezmi and Paskov, Kelley and Chrisman, Brianna and Stockham, Nate and Washington, Peter and Wall, Dennis},
  journal={arXiv preprint arXiv:2108.07917},
  year={2021}
}
```
