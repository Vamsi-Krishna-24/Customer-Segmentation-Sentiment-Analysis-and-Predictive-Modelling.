<!-- Project Title and Subtitle with Symbols -->
<h1>🎯 Customer Segmentation</h1>
<h2>📊 A Detailed Brief </h2>

<h3>Sentiment Analysis and Predictive Analysis</h3>
<!-- Brief with Image -->
<p>For any business to run in the most efficient and successful way, the primary aspect of that business should be customer retention. In other words, it's the ability of the company to turn customers into repeated buyers of their product or service.
For any Files related to data or code have a look on the left panel.</p>
<p>Companies effectively using customer segmentation strategies experience a 36% increase in revenue (Source: Aberdeen Group).</p>
<!-- Customer Segmentation Section with Image -->
<h3>🎯 Customer Segmentation</h3>
<img src="https://www.corporatevision-news.com/wp-content/uploads/2022/10/Customer-Segmentation.jpg" alt="Customer Segmentation" width="500"/>
<p>Within the vast array of reviews a company receives, we meticulously classify them into three distinct categories: positive, neutral, and negative. This segmentation enables us to effectively target positive, negative, and neutral customer segments with specific marketing strategies, maximizing the benefits for the company.</p>

<!-- Image -->
<img src="https://f.hubspotusercontent00.net/hubfs/285855/Compressed-bigstock-Customer-Retention-Strategy-D-255195535.jpg" alt="Customer Retention Strategy" width="500"/>

<!-- Data Collection Section -->
<h2>📈 Data Collection</h2>

<!-- Paragraph -->
<p>Nothing can be built from a void. Ever wondered why every supermarket, multiplex, airlines, any form of proper business asks for feedback? Is it just for improvising? Absolutely no, so what else is the question. Let's dive deep and see what else...</p>

The complete data that is dealt with in this project is [here](https://github.com/Vamsi-Krishna-24/Customer-Segmentation-Sentiment-Analysis-and-Predictive-Modelling./tree/main/Data_Folder).

There are two types of data:

1. **Customer Reviews Data:**
   - The first dataset consists of customer reviews. These reviews gathered as feedback on the company's website, have been web-scraped using a library called BeautifulSoup in Python. For a more detailed overview, you can find the [source code](https://github.com/Vamsi-Krishna-24/Customer-Segmentation-Sentiment-Analysis-and-Predictive-Modelling./blob/main/Data-Analysis-model-Building/Data_collectiion_through_web_scrapping.ipynb).
   - The actual [customer feedback](https://github.com/Vamsi-Krishna-24/Customer-Segmentation-Sentiment-Analysis-and-Predictive-Modelling./blob/main/Data_Folder/reviews.csv) that is collected through scraping. There are a total of 10,000+ records collected.

2. **Customer Booking Data:**
   - The second type of data is the customer booking data, which includes details about flight time, price, duration, etc.
   - You can access the customer booking data [here](https://github.com/Vamsi-Krishna-24/Customer-Segmentation-Sentiment-Analysis-and-Predictive-Modelling./blob/main/Data_Folder/customer_booking.csv). This dataset comprises 50,000+ records with 7+ main metrics.


<!-- Image -->
<img src="https://avinetworks.com/wp-content/uploads/2023/06/web-scraping-diagram.png" alt="Web Scraping Diagram" width="800"/>

<!-- Code Box -->
<pre>
<code>
import requests
from bs4 import BeautifulSoup
import pandas as pd
</code>
</pre>

<!-- Out of Code Box -->
<p>All the code for data collection is available <a href="https://github.com/Vamsi-Krishna-24/Customer-Segmentation-Sentiment-Analysis-and-Predictive-Modelling./blob/main/Data-Analysis-model-Building/Data_collectiion_through_web_scrapping.ipynb">here</a>.</p>

<!-- Sideheading and Paragraph -->
<h3>🤔 Understanding Different Possibilities: How the Data Can Be Used</h3>
<p>Every time feedback is given, it's molded to segment users so that each individual part is dealt with accordingly by the sales and marketing team.</p>

<!-- In-Depth Paragraph -->
<p>In-depth, understanding whether a customer will choose to engage with a company again hinges on the feedback they provide. Although the feedback process is inherently straightforward, its significance lies in our ability to predict a customer's likelihood of returning.</p>

<h2>🔍 The solution 📕</h2>

<!-- Sentiment Analysis Section -->
<h3>🔍 The Role of Data Analysis - Sentiment Analysis</h3>
<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="https://drive.google.com/uc?export=view&id=1zpCBlr3aGy1oMEtDvMbiX5hUnCMmcQAd" alt="Sentiment Analysis Image 1" width="400"/>
  <img src="https://drive.google.com/uc?export=view&id=1nRfyR8b-WwSsF12le46nURH1oAmIZAH4" alt="Sentiment Analysis Image 2" width="400"/>
</div>

<!-- Paragraph -->
<p>Manual classification of millions of customer reviews in real-time is an arduous task. This is where the TextBlob Python library becomes indispensable. Serving as a tool for processing textual data, TextBlob offers a simplified API for common natural language processing tasks, such as identifying parts of speech and noun phrases.</p>

<!-- In-Depth Paragraph -->
<p>Sentiment analysis, a critical aspect of this process, involves TextBlob capturing phrases and words and classifying them into positive, neutral, or negative categories. In essence, sentiment analysis serves as our means to discern the emotions and sentiments expressed by individual customers in their reviews, providing valuable insights for strategic decision-making.</p>

<!-- Python Requirements Section -->
<h3>🐍 Python Requirements</h3>

<!-- Code Box for requirements.txt -->
<pre>
<code> 
pandas
numpy
textblob
scikit-learn
</code>
</pre>

<!-- Out of Code Box -->
<p>To view the requirements.txt file  <a href="https://github.com/Vamsi-Krishna-24/Customer-Segmentation-Sentiment-Analysis-and-Predictive-Modelling./blob/main/requirements.txt">here</a>.</p>

<!-- Building a Predictive Model Section -->
<h3>🛠️ Building a Predictive Model</h3>

<!-- Paragraph -->
<p>Additional data is gathered and collected, including booking data (buying data). Along with the web scraping data, this collected data is used to build a predictive machine learning model. In this case, the model used is the random forest classifier, which can predict whether or not a given customer with given metrics will buy the product or service of the company.</p>

<!-- Model Image -->
<div style="display: flex; justify-content: space-between; align-items: center;">
  <!-- First Model Image -->
  <img src="https://serokell.io/files/vz/vz1f8191.Ensemble-of-decision-trees.png" alt="Random Forest Classifier Model" width="400"/>
  
  <!-- Second Model Image -->
  <img src="https://www.qualtrics.com/m/assets/wp-content/uploads/2020/05/1359555_PredictiveAnalytics_01_050222.png" alt="Predictive Analytics" width="400"/>
</div>

<!-- Additional Metrics and Accuracy Score -->
<p>Few important metrics taken as input for a good predicted output include: 'purchase_lead', 'length_of_stay', 'flight_duration' to predict the booking output. The model achieved an accuracy score of 80%.</p>
<!-- References Section -->
<h3>📚 References</h3>

<!-- Reference List -->
<ul>
  <li>Ref 1: <a href="https://www.airlinequality.com/" target="_blank">Airline Quality</a> - Website where data is webscrapped</li>
  <li>Ref 2: <a href="https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524" target="_blank">Towards Data Science</a> - Understanding TextBlob and NLP</li>
  <li>Ref 3: <a href="https://scikit-learn.org/stable/modules/model_evaluation.html" target="_blank">scikit-learn Documentation</a> - Understanding the models</li>
</ul>



