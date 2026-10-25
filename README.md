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


2. **Scroll down to "What did the hidden neurons learn?"** Pick one neuron whose picture looks
   like it has a clear shape and take a screenshot of it.



3. **In the Back-propagation tab, leave the learning rate at 0.50** and click **Next step ▶**
   through all 5 steps. Take a screenshot of step 2 (the error) and of step 5 (the weight table and
   the "Error after one update" number).



4. **In the Training Lab tab, train at least 4 networks** and keep the run-history table:
   - one with **all settings at their defaults** (you should get exactly the canonical network),
   - one with a learning rate that is **too low**,
   - one with a learning rate that is **too high**,
   - one with **"all zeros"** initial weights.

   Then experiment freely to find the best test accuracy you can. Take a screenshot of your run
   history table.


 

5. **In the Draw & Classify tab**, look up **test images #24, #4, and #6** with the
   "Test image number" box. For each one, write down the true label, the canonical network's
   prediction, and its confidence. Take a screenshot of image #6.

   

6. **Draw at least 3 digits of your own** on the canvas and note what the network answers each
   time. Take a screenshot of at least one attempt, including the small "What the network sees"
   image.



7. **Fill out `A15_ReflectionQuestions.md`**, using the exact numbers from your screenshots, and
   submit it along with your labeled screenshots (a `.zip` with the `.md` file and the images is
   fine).

 
---







# References:
- [Streamlit documentation](https://docs.streamlit.io/)
- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
