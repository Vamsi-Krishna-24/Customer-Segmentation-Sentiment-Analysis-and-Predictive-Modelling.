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
<p>Within the vast array of reviews a company receives, we meticulously classify them into three distinct categories: positive, neutral, and negative. This segmentation enables us to effectively target positive, negative, and neutral customer segments with specific marketing strategies, maximizing the benefits for the company.</p>

<!-- Image -->
<!-- Two Images Side by Side -->
<div style="display: flex;">
  <!-- First Image -->
  <img src="https://f.hubspotusercontent00.net/hubfs/285855/Compressed-bigstock-Customer-Retention-Strategy-D-255195535.jpg" alt="Customer Retention Strategy" width="500" style="margin-right: 10px;"/>

  <!-- Second Image -->
  <img src="https://www.corporatevision-news.com/wp-content/uploads/2022/10/Customer-Segmentation.jpg" alt="Customer Segmentation" width="500"/>
</div>


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
<h2>DATA PRE-PROCESSING</h2>


The available or collected data is often in a raw state, characterized by duplicate entries, null values, and a lack of structure. This raw state can be compared to chopping vegetables before cooking. To prepare the data for analysis, it requires cleaning and transformation:

- Removal of duplicate entries.
- Addressing null values.
- Changing the unstructured format to a structured one.

Just like preparing ingredients before cooking, this data cleaning process is essential before diving into the analysis or data analysis "recipe." The objective is to have a clean and organized dataset, ready for analysis—a necessary step before embarking on the analytical journey, much like having ingredients neatly prepared before starting to cook a delicious meal.


<h3>🤔 Understanding Different Possibilities: How the Data Can Be Used</h3>
<p>Every time feedback is given, it's molded to segment users so that each individual part is dealt with accordingly by the sales and marketing team.</p>

<!-- In-Depth Paragraph -->
<p>In-depth, understanding whether a customer will choose to engage with a company again hinges on the feedback they provide. Although the feedback process is inherently straightforward, its significance lies in our ability to predict a customer's likelihood of returning.</p>

<h2>🔍 In-detail DATA ANALYSIS and INSIGHTS 📕</h2>
<!-- 1. NLP and Sentiment Analysis -->
<table style="width: 100%;">
  <tr>
    <td style="width: 50%;">
      <p><strong>NLP and Sentiment Analysis:</strong> NLP and sentiment analysis play a crucial role in identifying customer satisfaction levels:</p>
      <ul>
        <li><strong>Satisfied Customers:</strong> 64%+</li>
        <li><strong>Unsatisfied Customers:</strong> 35%+</li>
        <li><strong>Neutral Responses:</strong> 11%</li>
      </ul>
    </td>
    <td style="width: 50%;">
      <img src="https://imgur.com/0QiAtT2.png" alt="NLP and Sentiment Analysis" width="500" />
      <br/><sub>↑ Explore NLP and Sentiment Analysis</sub>
    </td>
  </tr>
</table>

<!-- 2. Keyword Identification -->
<table style="width: 100%;">
  <tr>
    <td style="width: 50%;">
      <p><strong>Keyword Identification:</strong> Identifying the most dominant factors customers look for:</p>
      <ul>
        <li>Flight</li>
        <li>Food</li>
        <li>Service</li>
        <li>Trip Verified</li>
        <li>...</li>
      </ul>
    </td>
    <td style="width: 50%;">
      <img src="https://imgur.com/zs18jZ9.png" alt="Keyword Identification" width="500" />
      <br/><sub>↑ Discover Dominant Keywords</sub>
    </td>
  </tr>
</table>

<!-- 3. Derived Insight from the Other 2 Insights -->
<table style="width: 100%;">
  <tr>
    <td style="width: 50%;">
      <p><strong>Derived Insight from the Other 2 Insights:</strong> Predictions derived from the insights above:</p>
      <ul>
        <li>Transformed satisfaction levels.</li>
        <li>Predicted potential buyers with a precision score of 72%+.</li>
      </ul>
    </td>
    <td style="width: 50%;">
      <img src="https://imgur.com/8GvmVQU.png" alt="Derived Insight from the Other 2 Insights" width="500" />
      <br/><sub>↑ Explore Derived Insight</sub>
    </td>
  </tr>
</table>

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
<h2>🛠️ Building a Predictive Model</h2>

<!-- Paragraph -->
<p>Additional data is gathered and collected, including booking data (buying data). Along with the web scraping data, this collected data is used to build a predictive machine learning model. In this case, the model used is the random forest classifier, which can predict whether or not a given customer with given metrics will buy the product or service of the company.</p>

**Random Forest Classifier Usage:**

The Random Forest classifier is utilized by inputting data from the customer booking data matrix. This matrix incorporates features like flight time, hours traveled, and other relevant metrics. The primary goal is to predict whether the customer will make a repeat purchase from the company.

**Model Performance:**

The model's predictive performance is assessed using the Precision score, achieving an accuracy rate of 80%. This signifies a commendable level of precision in predicting positive instances, indicating a successful outcome.



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



