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

# Activity 15 — Reflection Questions: Digit Lab

Answer every question below using the **canonical network** and the app's default values, unless a
question specifically asks about your own Training Lab runs or drawings. Submit this file together
with your labeled screenshots — do not just describe the app, use your actual results.

---

## Part A — Representation (Session 25)

1. **How many neurons does the canonical network have in its **input layer**, its **hidden layer**,
   and its **output layer**? Explain why the input layer has exactly that many neurons, and why
   the output layer has exactly 10.**
   
- The canonical network has 784 input neurons, 64 hidden neurons, and 10 output neurons. The input layer contains 784 neurons because each MNIST image is 28 × 28 pixels, and every pixel becomes one numerical input for the network (28 × 28 = 784). The output layer contains 10 neurons because the network must classify digits from 0 to 9, so there is one output neuron for each possible digit.

  
2. **Report the canonical network's number of **trainable parameters**, and show how that number is
   computed from the layer sizes (weights + biases between each pair of layers).**
- The network has 50,890 trainable parameters.
- **Input - Hidden layer: **(784 × 64) + 64 = 50,240
-** Hidden - Output layer:** (64 × 10) + 10 = 650
- **Total:** 50,240 + 650 = 50,890 trainable parameters.

3. **Report the canonical network's **train accuracy, test accuracy, train loss, and test loss**.**

- Train accuracy: 97.2%
- Test accuracy: 94.1%
- Train loss: 0.106
- Test loss: 0.201


4. **Choose **one hidden neuron** from "What did the hidden neurons learn?" (include its screenshot).
   Describe what kind of shape or stroke it seems to respond to. Nobody programmed that pattern —
   so where did it come from?**
   
- I selected Neuron 10.
Its pattern shows a strong red vertical region that looks similar to part of a handwritten stroke, like the digit "1". The red areas indicate locations where ink strongly activates the neuron, while the blue areas indicate regions where ink suppresses its activation.

This pattern was not programmed manually. It emerged automatically during training through back-propagation. By seeing thousands of handwritten digits, the network adjusted its weights until this neuron became specialized in detecting a useful feature that helps distinguish digits.

---
## Part B — Back-propagation (Session 26)

5. **In the Back-propagation tab with the default learning rate of 0.50, report:
   (a) the network's output **o** before the update,
   (b) the error **E** before the update,
   (c) the new value of weight **w5** after the update, and
   (d) the error **after one update**.**

Using the default learning rate of 0.50, the network produced an output value (o) of 0.5411 before the update. The target value was 1.0, so the prediction was lower than desired.

- **The error before the update was: **E = 0.1053

During back-propagation, the gradients were calculated and used to update every weight in the network. After the update, weight w5 changed from 0.5000 to 0.5336.
As a result of the weight adjustments, the error decreased from 0.1053 to 0.0945. This shows that the update moved the network in a direction that improved its prediction and reduced the overall error.

(a) Output before update (o): 0.5411
(b) Error before update (E): 0.1053
(c) New value of weight w5 after update: 0.5336
(d) Error after one update: 0.0945

6. **In your own words, explain what the **backward pass** (steps 3 and 4) does and why the
   algorithm is called *back*-propagation. Use the deltas you saw (δo, δh1, δh2) in your
   explanation.**

The backward pass is the stage where the network learns from its mistakes. After the forward pass produces an output, the network calculates how far that prediction is from the correct answer. In this example, the output neuron produced o = 0.5411 when the target value was 1.0, resulting in an error of 0.1053.

The process begins with the output delta:

δo = -0.1140

This value measures how much the output neuron contributed to the error and indicates the direction in which the output should move to reduce that error.

The error is then propagated backward to the hidden layer. Each hidden neuron receives part of the responsibility for the mistake through its connection to the output neuron. The hidden deltas were:

δh1 = -0.0138

δh2 = 0.0169

These deltas show how much each hidden neuron contributed to the final error. Once the deltas are known, the network calculates the gradient for every weight. Each gradient determines how much that weight should be updated.

This process is called back-propagation because the error information flows backward through the network, from the output layer to the hidden layer. By repeatedly updating the weights using these gradients, the network gradually reduces its error and improves its predictions. In this example, the error decreased from 0.1053 to 0.0945 after a single update, demonstrating that the learning process was working correctly.

7. **Paste your Training Lab **run-history table** (at least 4 runs: defaults, a learning rate that
   is too low, one that is too high, and "all zeros" initial weights). For the too-low and too-high
   runs, describe what the loss and accuracy curves looked like and why that learning rate caused
   it.**
   
Run	Settings	Train Accuracy	Test AccuracyDefault (canonical)	64 hidden, sigmoid, LR=0.5, 10 epochs, batch=32, small random	97.2%	94.1%
Too Low LR	64 hidden, sigmoid, LR=0.001, 10 epochs, batch=32, small random	56.6%	55.4%
Too High LR	64 hidden, relu, LR=10.0, 10 epochs, batch=32, small random	10.0%	10.0%
All Zeros	64×64 hidden, relu, LR=0.5, 10 epochs, batch=64, all zeros	10.0%	10.0%

The default network achieved the best performance.

With the very low learning rate (0.001), the loss decreased slowly and the accuracy continued increasing at the final epochs. The network was learning, but it was learning too slowly.

With the very high learning rate (10.0), the network failed to learn and stayed near random-guess performance. The updates were too large, preventing the network from converging.


8. **What happened when you trained with **"all zeros"** initial weights? Why can't a network learn
   well if every weight starts at exactly the same value? (Think about what each hidden neuron
   computes when all of them have identical weights.)**
   
The network initialized with all-zero weights achieved only 10.0% train accuracy and 10.0% test accuracy.

This happens because all neurons start with identical weights and therefore compute exactly the same thing. During training, they receive identical updates and remain identical. Because the hidden neurons never develop different features, the network cannot learn effectively and performs only as well as random guessing.

9. **What was the **best test accuracy** you found, and with which settings? Did making the network
   bigger (more neurons or a second layer) or training longer (more epochs) *always* improve the
   test accuracy? Use the **train − test gap** from your runs to explain what you observed.**

**My best configuration was:
**
Hidden layers: 1
Neurons per hidden layer: 64
Activation function: sigmoid
Learning rate: 0.5
Epochs: 10
Batch size: 32
Initial weights: small random

**Results:**

Train accuracy: 97.2%
Test accuracy: 94.1%
Train-test gap: 3.1 points

Making the network bigger or training longer did not always improve test accuracy. A larger network can increase the difference between training accuracy and test accuracy, leading to overfitting. In my experiments, the default configuration produced the best test accuracy.

---
## Part C — Applications (Session 27)

10. **For test images **#24, #4, and #6**, report the true label, the canonical network's
    prediction, and its confidence. Then report how many of the 1,000 test images the canonical
    network gets wrong.**

Test image #24:

True label: 7
Prediction: 7
Confidence: 100.0%
Test image #4:

True label: 4
Prediction: 4
Confidence: 85.9%
Test image #6:

True label: 9
Prediction: 4
Confidence: 56.0%
The canonical network correctly classified test images #24 and #4 with high confidence. However, it misclassified image #6. The true label was 9, but the network predicted 4 with 56.0% confidence.

This mistake shows that neural networks do not actually "understand" digits the way humans do. Instead, they recognize patterns that resemble examples seen during training. In image #6, the handwritten 9 contains strokes that visually resemble a 4, which caused the network to assign a higher probability to the wrong class.

11. **Describe the results of your **own drawings** (at least 3): which digits did you draw, and
    what did the network answer each time? Give at least one reason why your drawings can be
    harder for this network than the MNIST test images, even though they look perfectly clear to a
    human.**

Drawing 1:

Digit drawn: 2
Network prediction: 2
Drawing 2:

Digit drawn: 7
Network prediction: 7
Drawing 3:

Digit drawn: 9
Network prediction: 4
The network correctly identified the first two drawings but struggled with the third one. Even though the digit looked clear to me, its shape was slightly different from the examples used during training.

My drawings can be harder for the network than the MNIST test images because MNIST digits are carefully centered, normalized, and scaled to a consistent 28 × 28 format. Personal drawings may be off-center, have different stroke thicknesses, unusual handwriting styles, missing segments, or extra marks.

Humans can usually recognize digits despite these variations because we use context and prior experience. A neural network, however, relies entirely on patterns learned from its training data, so unusual drawings can lead to incorrect predictions.

12. **Choose **one application area** from Session 27 (for example medical diagnosis, industrial
    quality control, autonomous driving, finance, agriculture, or education). Describe a neural
    network for it: what would its **inputs** be, what would its **outputs** be, and what is one
    real risk if it makes a wrong prediction, like the canonical network does on image #6?**

One important application of neural networks is reading handwritten postal codes on envelopes and packages.

The input to the network would be an image containing handwritten digits. The output would be the predicted postal code, represented as a sequence of recognized numbers.

This application is very similar to the digit-recognition task performed in this activity. Just as the network attempted to recognize the digits in the MNIST dataset, a postal-service system must recognize handwritten numbers written by thousands of different people.

A real risk occurs when the network makes a mistake. For example, in test image #6 the network confused a handwritten 9 with a 4. If the same type of error happened while reading a postal code, a package or letter could be delivered to the wrong destination, causing delays, increased costs, or lost mail. This demonstrates why accuracy is important and why neural-network predictions should sometimes be verified when the confidence is low.




# References:
- [Streamlit documentation](https://docs.streamlit.io/)
- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
