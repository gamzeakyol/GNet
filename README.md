# GNet

### A Variational Graph Autoencoder for Manipulation Action Recognition and Prediction
_Gamze Akyol, Sanem Sariel, Eren Erdal Aksoy_  
[PDF link](https://arxiv.org/abs/2110.13280#)

_Abstract— Despite decades of research, understanding human manipulation activities is, and has always been, one of the most attractive and challenging research topics in computer vision and robotics. Recognition and prediction of observed human manipulation actions have their roots in the applications related to, for instance, human-robot interaction and robot learning from demonstration. The current research trend heavily relies on advanced convolutional neural networks to process the structured Euclidean data, such as RGB camera images. These networks, however, come with immense computational complexity to be able to process high dimensional raw data. Different from the related works, we here introduce a deep graph autoencoder to jointly learn recognition and prediction of manipulation tasks from symbolic scene graphs, instead of relying on the structured Euclidean data. Our network has a variational autoencoder structure with two branches: one for identifying the input graph type and one for predicting the future graphs. The input of the proposed network is a set of semantic graphs which store the spatial relations between subjects and objects in the scene. The network output is a label set representing the detected and predicted class types. We benchmark our new model against different state-of-the-art methods on two different datasets, MANIAC and MSRC-9, and show that our proposed model can achieve better performance. We also release our source code https://github.com/gamzeakyol/GNet._

![GraphNet_yeni_decodersiz_(6)](https://user-images.githubusercontent.com/15743753/138309567-6a6abadf-4487-4b16-b745-0a236c634928.png)


## Dataset
![KeyFrame_005850 (2)](https://user-images.githubusercontent.com/15743753/148391329-0a61cd02-f872-42aa-8251-b0b4e22687a5.png)
![KeyFrame_005959 (2)](https://user-images.githubusercontent.com/15743753/148391361-adb89525-ff0a-4220-878f-43637c199a14.png)
![KeyFrame_000438 (2)](https://user-images.githubusercontent.com/15743753/148391371-5ab95005-7340-40df-ad95-2a7df0481284.png)
![KeyFrame_000705_orange](https://user-images.githubusercontent.com/15743753/148391376-3d8e7a49-17f6-44a7-b784-75cd5d7a523e.png)
Sample frames from different action classes in the MANIAC dataset (Aksoy et. al., 2015). The top and bottom rows show the pushing and stirring actions, respectively. Each color here represents a unique image segment (i.e., an object), from which graphs are derived. Relations between “touching” segments are shown with the blue edges.


### Citation

Accepted for publication at 20th International Conference on Advanced Robotics (ICAR).  

@misc{akyol2021variational,  
    title={A Variational Graph Autoencoder for Manipulation Action Recognition and Prediction},  
    author={Gamze Akyol and Sanem Sariel and Eren Erdal Aksoy},  
    year={2021},  
    eprint={2110.13280},  
    archivePrefix={arXiv},  
    primaryClass={cs.CV}  
}
