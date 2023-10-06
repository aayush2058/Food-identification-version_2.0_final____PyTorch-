
# 1.
import gradio as gr
import os
import torch

from model import create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open("class_names.txt", "r") as f:
    class_names = [food.strip('\n') for food in f.readlines()]

# 2.
vit_food101, vit_transforms = create_vit_model(num_classes = 101)

# Load saved weights
vit_food101.load_state_dict(torch.load("models/state_dict__vit_food101_100_percent.pth",
                                           map_location = torch.device('cpu')))


# 3.
def predict(img) -> Tuple[Dict, float]:
    
    # Start a timer
    start_time = timer()
    
    # Transform the input image for use with vit
    img = vit_transforms(img).unsqueeze(0)
    
    # Put the model into eval mode, make prediction
    vit_food101.eval()
    with torch.inference_mode():
        # Pass transformed image through the model and turn the prediction logits into probabilities
        pred_probs = torch.softmax(vit_food101(img), dim = 1)
    
    # Create a prediction label and prediction probability dictionary
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate pre time
    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    
    # Return pred dict and pred time
    return pred_labels_and_probs, pred_time

# 4.
title = 'FoodIdentifier Big 💪🍕'
description = "A Vision Transformer feature extractor computer vision model to classify images as pizza, sushi or steak"
article = " anything I want for the description of the description above 🤪"

# Create example list
# Get example filepaths in a list of lists
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn = predict, # maps input to output
                    inputs = gr.Image(type = 'pil'),
                    outputs = [gr.Label(num_top_classes = 5, label = "Predictions"),
                              gr.Number(label = "Prediction time (s)")],
                    examples = example_list,
                    title = title,
                    description = description,
                    article = article
                   )

# Launch the demo
demo.launch(debug = False, # print errors locally?
           share = True) # generate a publically shareable URL
