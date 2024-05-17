<!DOCTYPE html>
<html>
<head>
</head>
<body>

<h1>BayMax- A Medical Assistance Bot</h1>

<p>This repository contains a Python-based project aimed at creating a small, fast, and efficient medical assistance bot that provides better assistance than voice command help from Google. The project uses a  Random forest classifier for diagnosis and a DNN using BOW ( bag of words ) trained on an intents json file. The project leveraged a LLaMA 3B 7B 8-bit quantized model to prompt engineer as to generate missing responses, and create extra strings for pattern matching to increase the accuracy of disease classification in the JSON dataset DNN. Additionally, I scraped disease classifications from the classification dataset and generated tags, strings, patterns, and responses to add to the intents file in order for it to also be able to provide what the medicines and treatments may be like after diagnosis is done.
<p align="center">
    <img src="https://github.com/samraatz/BayMax-Mini/blob/main/home.png" alt="home page width="500"></img>
</p>
</p>
<h2>Table of Contents</h2>
<ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
</ul>

<h2 id="overview">Overview</h2>
<p>The goal of this project is to develop a medical assistance bot that is compact, rapid, and offers superior assistance compared to existing voice command help systems like Google. The project synthetically augments the dataset using a LLaMA 3B 7B 8-bit quantized model, ensuring comprehensive coverage of potential disease classifications and their associated responses.</p>
<p>Below is an example of how where google fails, our model succeeds.</p>
<div class="container">
    <div class="row">
        <div class="column">
            <img src="https://github.com/samraatz/BayMax-Mini/blob/main/g1.jpeg" alt="google query" width="250">
        </div>
        <div class="column">
            <img src="https://github.com/samraatz/BayMax-Mini/blob/main/g2.jpeg" alt="google response" width="250">
        </div>
    </div>
    <div class="full-width">
        <img src="https://github.com/samraatz/BayMax-Mini/blob/main/first%20aid-response.png" alt="our response" width="500">
    </div>
</div>
<p>The intent dataset was initially sourced from <a href="https://www.kaggle.com/datasets/therealsampat/intents-for-first-aid-recommendations/data">Kaggle-FirstAidIntents</a>. This dataset had many empty response lists in tags and a low number of strings for pattern matching. It also did not include the diseases we intended to diagnose.</p>
    <img src="https://github.com/samraatz/BayMax-Mini/blob/main/original%20datset%20-%20empty%20responses%20and%20no%20tags%20for%20diseases.png" alt="original dataset" width="500"></img>

<p>The classification dataset was sourced from <a href="https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning">Kaggle-Diseases</a>. TThe avaialable classifications were scraped, and added by generating them as tags for the intents data.</p>

<h2 id="features">Methodology and Features</h2>
<ul>
    <li><strong>Synthetic Augmentation</strong>: Used a LLaMA 3B 7B 8-bit quantized model to generate missing responses and create additional strings for pattern matching.<img src="https://github.com/samraatz/BayMax-Mini/blob/main/Data%20Augmentation%20-%201.png" alt="generation" width="500"></img>
</li>
    <li><strong>Disease Classification Scraping</strong>: Extracted disease classifications from the dataset, used llama 3 to create accurate tags and responses.</li>
    <img src="https://github.com/samraatz/BayMax-Mini/blob/main/augmented%20strings.png" alt="new strings" width="500"></img>
    <li><strong>Pattern Matching</strong>: Implements advanced pattern matching techniques to increase classification accuracy.</li>
    <li><strong>Response Generation</strong>: Generates relevant and accurate responses for the bot based on classified diseases.</li>
    <li><strong>Diagnosis and First Aid Options</strong>: Allows the user to choose between diagnosing diseases or receiving first aid recommendations.</li>
</ul>

<h2 id="installation">Installation</h2>
<p>To install and set up the project, follow these steps:</p>
<ol>
    <li><strong>Clone the repository:</strong></li>
    <pre><code>git clone https://github.com/yourusername/medical-assistance-bot.git
cd medical-assistance-bot</code></pre>
    <li><strong>Create a virtual environment and activate it:</strong></li>
    <pre><code>python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`</code></pre>
    <li><strong>Install the required dependencies:</strong></li>
    <pre><code>pip install -r requirements.txt</code></pre>
</ol>

<h2 id="usage">Usage</h2>
<p>To use the project, follow these steps:</p>
<ol>
    <li><strong>Start the main application:</strong></li>
    <pre><code>streamlit run workingvoice.py</code></pre>
    <p>The application will prompt the user to choose between diagnosing a disease or getting first aid help.</p>
    <li><strong>For disease diagnosis:</strong></li>
    <ul>
        <li>Speak out diagnosis and type the symptoms separated by commas (e.g., "fever, cough, headache")            <img src="https://github.com/samraatz/BayMax-Mini/blob/main/diagnosis.png" alt="diagnosis home" width="500"></img>
.</li>
        <li>The Random Forest classifier will classify the disease based on the symptoms.</li>
        <li>The classification result will be spoken out loud.<img src="https://github.com/samraatz/BayMax-Mini/blob/main/diagnosis-response-1.png" alt="diagnosis 1" width="500"></img></li>
        <li>Speak the diagnosed disease, which will be matched to the intents in the final intent JSON file using the DNN model.</li>
        <li>A response including recommended medicines and treatment will be generated and spoken out.<img src="https://github.com/samraatz/BayMax-Mini/blob/main/diagnosis-response-3.png" alt="diagnosis response" width="500"></img></li>
    </ul>
    <li><strong>For first aid help:</strong></li>
    <ul>
        <li>Speak your issue or symptom.</li>
        <li>The system will match your input to the most plausible response using a Bag of Words model (implemented with NLTK) and a DNN.</li>
        <li>The appropriate first aid recommendation will be provided.<img src="https://github.com/samraatz/BayMax-Mini/blob/main/first%20aid-response.png" alt="first aid response" width="500"></img></li>
    </ul>
</ol>

<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.</p>
<ol>
    <li><strong>Fork the repository</strong></li>
    <li><strong>Create a new branch for your feature/bugfix</strong></li>
    <li><strong>Commit your changes</strong></li>
    <li><strong>Push to the branch</strong></li>
    <li><strong>Create a pull request</strong></li>
</ol>

<h2 id="license">License</h2>
<p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
