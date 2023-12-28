<!-- Project Title and Subtitle with Symbols -->
<h1>🎯 Customer Segmentation</h1>
<h2>📊 1.A Detailed Brief </h2>

<h3>1.1Sentiment Analysis and Predictive Analysis</h3>
<!-- Brief with Image -->
<p>For any business to run in the most efficient and successful way, the primary aspect of that business should be customer retention. In other words, it's the ability of the company to turn customers into repeated buyers of their product or service.
For any Files related to data or code have a look on the left panel.</p>
<p>Companies effectively using customer segmentation strategies experience a 36% increase in revenue (Source: Aberdeen Group).</p>
<!-- Customer Segmentation Section with Image -->
<h3>1.2🎯 Customer Segmentation</h3>
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
<h2>2. Data Collection 📈</h2>

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
<h2>3.DATA PRE-PROCESSING</h2>
<p>The term "Pre-processing" itself says that data is handled "PRE" (before) processing into the code. It has two main steps Cleaning and feature engineering. </p>
<h3>3.1 Data Cleaning</h3>
The available or collected data is often in a raw state, characterized by duplicate entries, null values, and a lack of structure. This raw state can be compared to chopping vegetables before cooking. To prepare the data for analysis, it requires cleaning and transformation:

- Removal of duplicate entries.
- Addressing null values.
- Changing the unstructured format to a structured one.

<h3>3.2 Feature Engineering.</h3>
From the data that is available in hand, creating new metrics(columns) or new Features in data, results in analyzing the data in the required state.

Just like preparing ingredients before cooking, this data-cleaning process is essential before diving into the analysis or data analysis "recipe." The objective is to have a clean and organized dataset, ready for analysis—a necessary step before embarking on the analytical journey, much like having ingredients neatly prepared before starting to cook a delicious meal.
Here a new feature is added that is **SENTIMENT** after performing sentiment analysis, to study the percentage of customers who stays **Positive, Negative and Neutral** towards a service.
<!-- Code Box -->
<pre>
<code>
# Apply sentiment analysis and categorization to each comment
data['Sentiment'] = data['Reviews'].apply(lambda x: categorize_sentiment(analyse_sentiment(x)))
</code>
</pre>

<h3>3.3 Understanding Different Possibilities: How the Data Can Be Used🤔</h3>
<p>Every time feedback is given, it's molded to segment users so that each individual part is dealt with accordingly by the sales and marketing team.</p>


<!-- In-Depth Paragraph -->
<p>In-depth, understanding whether a customer will choose to engage with a company again hinges on the feedback they provide. Although the feedback process is inherently straightforward, its significance lies in our ability to predict a customer's likelihood of returning.</p>

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


<h2>4. In-detail DATA ANALYSIS and INSIGHTS 🔍📕</h2>
<!-- 1. NLP and Sentiment Analysis -->
<table style="width: 100%;">
  <tr>
    <td style="width: 50%;">
      <p><strong>4.1:NLP and Sentiment Analysis:</strong> NLP and sentiment analysis play a crucial role in identifying customer satisfaction levels:</p>
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
      <p><strong>4.2:Keyword Identification:</strong> Identifying the most dominant factors customers look for:</p>
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

<!-- 3.Flight booking on each day -->
<table style="width: 100%;">
  <tr>
    <td style="width: 50%;">
      <p><strong>4.3:Daily basis flight booking data </strong> </p>
      <ul>
        <li>Weekly Decline in Booking .</li>
        <li>Highest booking on Monday and gradually decrease by 5% every day. </li>
        <li>After the fall of bookings from Monday to Saturday again there is a slight raise that occurs on sunday</li>
      </ul>
    </td>
    <td style="width: 50%;">
      <img src="https://imgur.com/kUZDbiM.png" alt="Daily Flight booking data" width="500" />
      <br/><sub>↑ Flight booking data on daily basis</sub>
    </td>
  </tr>
</table>

<h2>5.End-to-end machine learning pipeline or workflow</h2>

<h3>5.1 S3 Bucket Creation and Data Upload: 🪣</h3>
S3 buckets are like digital containers in the cloud that store various types of data securely. They act as virtual warehouses accessible from anywhere on the internet, offering features like versioning, access control, and the ability to host static websites. 
In simple terms let us say photos in our mobile is synced into google Photos, thereby managing the storage and can be accessed from anywhere.The simple logic applies here to...

<img src="https://imgur.com/ew8nFid.png" alt="S3 Buckets" width="900">

<h3>5.2 Data Access through SageMaker:</h3> 
In the machine learning workflow, data stored in an S3 bucket is seamlessly accessed through SageMaker, where a Jupyter Notebook is employed for developing and fine-tuning the machine learning model. For instance, consider a scenario where a dataset of housing prices (stored in S3) is analyzed and a predictive model is trained using SageMaker's Jupyter environment for housing price predictions.
<img src="https://imgur.com/s9aOlk6.png"  width="900">

<h3>5.3 Machine Learning Code Development in Jupyter Notebook:</h3> 

S3 stores data for ML in a bucket, accessed by SageMaker's Jupyter. For instance, predicting house prices using a dataset stored in S3 via SageMaker's Jupyter.
<img src="https://imgur.com/jA4n64x.png" width="900">

<h3>5.4 Deployment on EC2 Instance:</h3> 

After developing a sentiment analysis model in a Jupyter Notebook within Amazon SageMaker, you deploy the model as a service. This deployed model becomes accessible through an API endpoint, integrated seamlessly with an application running on an Amazon EC2 instance. For instance, a social media platform could utilize this pipeline to automatically analyze user sentiments in real-time comments, enhancing user experience.
<img src="https://imgur.com/Z7Ak5NP.png"  width="900">

<!-- Building a Predictive Model Section -->
<h2>6. 🛠️ Building a Predictive Model</h2>
<p>The core of this project lies at building an model and that's here....</p>
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
<!-- Conclusion Section -->
<h2>Conclusion</h2>

<!-- Conclusion Paragraph -->
<p>After meticulously collecting and analyzing customer feedback, it's evident that customers fall into two distinct types: those who actively provide feedback (Type 1) and those who do not (Type 2). The essence of this project revolves around understanding customer behavior and satisfaction levels.</p>

<p>The primary objective is to segment customers based on their satisfaction levels. Each segment is then channeled into specific marketing strategies, ensuring a tailored approach to maximize the chances of customers returning for the same product or service. By studying and responding to customer feedback, businesses can enhance customer satisfaction and loyalty, ultimately contributing to their overall success.</p>



