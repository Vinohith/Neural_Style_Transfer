# Neural Style Transfer

Neural Style Transfer is a method that allows us to generate Artistic Images by transfering the style of one image (say a beautiful painting) to another image (say a photo of a person). The resulting output is one image in the style of another. 

This repository contains the code to run the model for Style Transfer on any images of ones choice. The model used is the VGG-19 architecture trained on ImageNet images. 
The input to the network are the **Content/Base Image(C)**, the primary image on which the style is to be transfered and the **Style Image(S)**, the image whose style is to be transfered.

# Content Image
<img width="400" height="400" src="Content_Image/IMG-20170808-WA0000.jpg"/>

# Style Image and Generated Image

<img width="400" height="400" src="Style_Images/water.png"/>    <img width="400" height="400" src="Generated_Images/water.gif"/>

<img width="400" height="400" src="Style_Images/saatchi_art.jpg"/>    <img width="400" height="400" src="Generated_Images/saatchi_art.gif"/>

<img width="400" height="400" src="Style_Images/mountain.png"/>    <img width="400" height="400" src="Generated_Images/mountain.gif"/>

<img width="400" height="400" src="Style_Images/sun_fog.png"/>    <img width="400" height="400" src="Generated_Images/sun_fog.gif"/>
