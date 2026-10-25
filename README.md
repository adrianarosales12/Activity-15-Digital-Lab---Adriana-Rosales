# Activity 15: Digit Lab — Teaching a Neural Network to Read Handwriting
## Sessions 25–27
## Due date (mm/dd/yyyy): 10/25/2026
## Adriana Rosales González
## Delivery Format: [] Video URL | [X] Markdown file | [] Jupyter Notebook file

---

# Activity Description

## The Story

Before you deposit a check from your phone, a computer reads the handwritten amount. Before a
letter reaches your door, a machine reads the handwritten postal code. Both jobs are done by
**neural networks** — programs loosely inspired by the neurons in your brain, which *learn* to
recognize patterns from thousands of examples instead of being given rules.

In this activity you get your own neural network that reads handwritten digits (0–9). You will
look inside it (**Session 25**: how a network represents an image), watch it learn one step at a
time (**Session 26**: back-propagation), train your own versions by changing the settings, and
finally test it on your own handwriting (**Session 27**: neural networks applied to a real problem).


### Your Tasks

1. **In the Meet the Network tab**, study the canonical network: its layers, its number of
   trainable parameters, its train/test accuracy and loss, and its training curves. Take a
   screenshot of the diagram and the five numbers below it.
<img width="617" height="613" alt="Captura de pantalla 2026-10-25 084855" src="https://github.com/user-attachments/assets/8574871b-cb04-4fff-9985-21f1777cdac2" />

<img width="626" height="237" alt="Captura de pantalla 2026-10-25 085501" src="https://github.com/user-attachments/assets/20b9a2c9-d789-4039-bbca-9a6ba6d7dc65" />


   - The canonical network has 784 input neurons, 64 hidden neurons, and 10 output neurons. The input layer contains 784 neurons because each MNIST image is 28 × 28 pixels, and every pixel becomes one numerical input for the network (28 × 28 = 784). The output layer contains 10 neurons because the network must classify digits from 0 to 9, so there is one output neuron for each possible digit.

**TRAINABLE PARAMETERS**   
   - The network has 50,890 trainable parameters.
   - Input - Hidden layer: (784 × 64) + 64 = 50,240
   - Hidden - Output layer: (64 × 10) + 10 = 650
   - Total: 50,240 + 650 = 50,890 trainable parameters.

**CANONICALY ACCURACY AND LOSS**
   - Train accuracy: 97.2%
   - Test accuracy: 94.1%
   - Train loss: 0.106
   - Test loss: 0.201


---

2. **Scroll down to "What did the hidden neurons learn?"** Pick one neuron whose picture looks
   like it has a clear shape and take a screenshot of it.
<img width="212" height="289" alt="image" src="https://github.com/user-attachments/assets/d61dc28f-5c90-48f0-9998-67dbdd9b6e53" />

- I selected Neuron 10. The red areas form a vertical stroke that looks similar to part of the digit "1". The red regions indicate places where ink makes the neuron activate strongly, while the blue regions indicate places where ink makes the neuron stay less active. This pattern was not programmed manually. It was learned automatically during training through back-propagation, as the network adjusted its weights to recognize useful features present in handwritten digits.


---
3. **In the Back-propagation tab, leave the learning rate at 0.50** and click **Next step ▶**
   through all 5 steps. Take a screenshot of step 2 (the error) and of step 5 (the weight table and
   the "Error after one update" number).

- **STEP 2 : ERROR**
<img width="570" height="607" alt="image" src="https://github.com/user-attachments/assets/79799bb5-d99c-4b71-b876-1c19f07c9efe" />

- **STEP 5: ERROR AFETER ONE UPTADE**
<img width="574" height="529" alt="image" src="https://github.com/user-attachments/assets/428c1038-d14b-406d-a013-5077d2067998" />
<img width="583" height="478" alt="image" src="https://github.com/user-attachments/assets/2a9d854c-8a4d-4e57-8400-68caf58e3596" />


---
4. **In the Training Lab tab, train at least 4 networks** and keep the run-history table:
   - one with **all settings at their defaults** (you should get exactly the canonical network),
   - one with a learning rate that is **too low**,
   - one with a learning rate that is **too high**,
   - one with **"all zeros"** initial weights.

   Then experiment freely to find the best test accuracy you can. Take a screenshot of your run
   history table.

<img width="551" height="628" alt="Captura de pantalla 2026-10-25 093522" src="https://github.com/user-attachments/assets/ad116fcd-11be-4d5e-8cc8-74944ec22ff2" />
<img width="565" height="616" alt="Captura de pantalla 2026-10-25 093317" src="https://github.com/user-attachments/assets/ce6dc172-0eff-4103-a01b-b3217b6cdd8e" />
<img width="563" height="629" alt="Captura de pantalla 2026-10-25 093231" src="https://github.com/user-attachments/assets/f9f2e37f-e80a-454b-aea1-24005bb6d6e8" />
<img width="558" height="637" alt="Captura de pantalla 2026-10-25 093606" src="https://github.com/user-attachments/assets/6cf215a3-ab9c-4318-9fd9-99f6e7531c83" />

- The default network achieved 94.1% test accuracy and 97.2% train accuracy.

The network with a very low learning rate (0.001) achieved 55.4% test accuracy and 56.6% train accuracy. The accuracy was still increasing at the end of training, showing that the network was learning too slowly and needed more epochs.

The network with a very high learning rate (10.0) achieved only 10.0% test accuracy and 10.0% train accuracy. The updates were too large, preventing the network from converging and causing it to perform no better than random guessing.

The default configuration produced the best performance among these required experiments.

   -**My best configuration was:**

Hidden layers: 1
Neurons per hidden layer: 64
Activation function: sigmoid
Learning rate: 0.5
Epochs: 10
Batch size: 32
Initial weights: small random

   -**Results:**
Train accuracy: 97.2%
Test accuracy: 94.1%
Train-test gap: 3.1 points
Bigger or longer networks do not always perform better. The train-test gap can increase when a network becomes too complex or trains for too long, which may lead to overfitting. In my experiments, the default configuration achieved the best test accuracy.

 
---
5. **In the Draw & Classify tab**, look up **test images #24, #4, and #6** with the
   "Test image number" box. For each one, write down the true label, the canonical network's
   prediction, and its confidence. Take a screenshot of image #6.
<img width="552" height="433" alt="Captura de pantalla 2026-10-25 094542" src="https://github.com/user-attachments/assets/1f8c24bf-63fd-4bc1-ae77-b6cd2a30f783" />
<img width="560" height="437" alt="Captura de pantalla 2026-10-25 094530" src="https://github.com/user-attachments/assets/f952a301-7acb-43f5-8f80-a7b69f0266ac" />
<img width="560" height="433" alt="Captura de pantalla 2026-10-25 094512" src="https://github.com/user-attachments/assets/b86bb822-9c11-4cb4-97c6-dcb60b2c1c2d" />

Image #24

True label: 7
Prediction: 7
Confidence: 100.0%
Image #4

True label: 4
Prediction: 4
Confidence: 85.9%
Image #6

True label: 9
Prediction: 4
Confidence: 56.0%
The canonical network misclassified image #6. The true label was 9, but the network predicted 4 because the handwritten digit looks similar to a 4.

Number of misclassified test images: (copy the number shown in the “Where the canonical network gets it wrong” section).
   
---
6. **Draw at least 3 digits of your own** on the canvas and note what the network answers each
   time. Take a screenshot of at least one attempt, including the small "What the network sees"
   image.

- NUMBER 2
<img width="584" height="617" alt="image" src="https://github.com/user-attachments/assets/4f8dee5c-438c-458c-8c55-185034658353" />

- NUMBER 7

  <img width="580" height="484" alt="image" src="https://github.com/user-attachments/assets/27b2c283-157a-46ab-8c2e-8975961e4d83" />

- NUMBER 9
<img width="576" height="427" alt="image" src="https://github.com/user-attachments/assets/2a2c3d91-2d32-409b-a063-95c3fd251de0" />


**Drawing 1:**

Drawn digit: 2
Network prediction: 2
Drawing 2:
**
Drawn digit: 7**
Network prediction: 7
Drawing 3:

**Drawn digit: 9**
Network prediction: 4
My drawings can be harder to classify than the MNIST test images because they may be off-center, have different stroke thicknesses, contain noise, or be written in a style that differs from the training data. The network was trained on standardized handwritten digits, while personal drawings may vary significantly in shape and position.

---
7. **Fill out `A15_ReflectionQuestions.md`**, using the exact numbers from your screenshots, and
   submit it along with your labeled screenshots (a `.zip` with the `.md` file and the images is
   fine).

 
---







# References:
- [Streamlit documentation](https://docs.streamlit.io/)
- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
