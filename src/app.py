import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import base64

# -------------------------
# DOWNLOAD VADER
# -------------------------
nltk.download('vader_lexicon')

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Rural Development Sentiment Dashboard",
    layout="wide"
)

# -------------------------
# BACKGROUND IMAGE
# -------------------------

from pathlib import Path

def get_base64(file):
    path = Path(__file__).parent.parent / file
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("background.jpg.jpeg")

page_bg = f"""
<style>

.block-container {{
padding-top: 3rem;
}}


.stApp {{
background: linear-gradient(rgba(255,255,255,0.75), rgba(255,255,255,0.75)),
url("data:image/jpg;base64,{bg_image}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
background: transparent;
}}

.stButton>button {{
background-color:#2D6A4F;
color:white;
border-radius:8px;
height:45px;
font-weight:600;
}}

.stButton>button:hover {{
background-color:#1B4332;
}}

textarea {{
border-radius:10px !important;
border:1px solid #d3d3d3 !important;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------

st.markdown("""
<h1 style='text-align:center; color:#1B4332; font-size:46px'>
🌾 AI-Driven Social Media Sentiment Analysis for Rural Development Initiatives
</h1>

<p style='text-align:center; font-size:20px; color:#444'>
Transforming Social Media Conversations into Rural Development Insights
</p>
""", unsafe_allow_html=True)

st.divider()

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "rural_tweets_dataset_10000.csv"
    df = pd.read_csv(data_path)

    if "tweet" in df.columns:
        df["post"] = df["tweet"].astype(str).str.lower()

    return df

df = load_data()

# -------------------------
# RURAL FILTER
# -------------------------

rural_keywords = [
    "farmer","farm","crop","irrigation","water","village",
    "agriculture","fertilizer","drought","harvest",
    "road","electricity","internet","subsidy","insurance",
    "soil","tractor","seed","market","mandi"
]

df_rural = df[df["post"].apply(
    lambda x: any(word in x for word in rural_keywords)
)]

# -------------------------
# AI SENTIMENT (Improved)
# -------------------------

sia = SentimentIntensityAnalyzer()

@st.cache_data
def analyze_sentiment(data):

    def get_sentiment(text):

        text = text.lower()

        negation_words = ["not","no","never","hardly","barely"]
        positive_words = ["help","helped","improve","improved","benefit","increase","growth"]

        # detect negation + positive phrase
        if any(n in text for n in negation_words) and any(p in text for p in positive_words):
            return "Negative"

        score = sia.polarity_scores(text)["compound"]

        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    data["sentiment_ai"] = data["post"].apply(get_sentiment)

    return data

df_rural = analyze_sentiment(df_rural)

# -------------------------
# SIDEBAR
# -------------------------

section = st.sidebar.selectbox(
    "Select Section",
    ["Sentiment Lab", "Dashboard", "Topic & Problem Analysis"]
)

# -------------------------
# SENTIMENT LAB
# -------------------------

if section == "Sentiment Lab":

    st.header("🧪 Sentiment Lab")

    user_input = st.text_area("Enter text")

    if st.button("Analyze Sentiment"):

        text = user_input.lower()

        negation_words = ["not","no","never","hardly","barely"]
        positive_words = ["help","helped","improve","improved","benefit","increase","growth"]

        if any(n in text for n in negation_words) and any(p in text for p in positive_words):
            sentiment = "Negative 😞"
        else:

            score = sia.polarity_scores(text)["compound"]

            if score > 0.05:
                sentiment = "Positive 😊"
            elif score < -0.05:
                sentiment = "Negative 😞"
            else:
                sentiment = "Neutral 😐"

        st.success(f"Sentiment: {sentiment}")

# -------------------------
# DASHBOARD
# -------------------------

elif section == "Dashboard":

    st.header("📊 Rural Sentiment Dashboard")

    num_posts = st.number_input(
        "Enter number of posts",
        min_value=10,
        max_value=len(df_rural),
        value=50
    )

    data = df_rural.head(num_posts)

    sentiment_counts = data["sentiment_ai"].value_counts()

    positive = sentiment_counts.get("Positive",0)
    neutral = sentiment_counts.get("Neutral",0)
    negative = sentiment_counts.get("Negative",0)

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("📄 Total Posts",num_posts)
    col2.metric("😊 Positive",positive)
    col3.metric("😐 Neutral",neutral)
    col4.metric("😞 Negative",negative)

    st.subheader("Top Rural Posts")

    st.dataframe(
        data[["post","sentiment_ai"]],
        use_container_width=True,
        hide_index=True
    )

    col1,col2 = st.columns(2)

    labels = ["Positive","Neutral","Negative"]
    values = [positive,neutral,negative]

    colors = ["#2D6A4F","#ADB5BD","#D00000"]

    # LINE GRAPH
    with col1:

        st.subheader("Rural Sentiment Trend")

        fig,ax = plt.subplots()

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.plot(labels,values,
                marker="o",
                linewidth=4,
                color="#2D6A4F")

        ax.set_xlabel("Sentiment Type")
        ax.set_ylabel("Number of Posts")

        ax.grid(True,linestyle="--",alpha=0.6)

        for i,v in enumerate(values):
            ax.text(i,v+1,str(v),ha='center')

        st.pyplot(fig)

    # DONUT CHART
    with col2:

        st.subheader("Rural Sentiment Distribution")

        fig2,ax2 = plt.subplots()

        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")

        ax2.pie(values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                wedgeprops={"width":0.4})

        st.pyplot(fig2)

# -------------------------
# TOPIC ANALYSIS
# -------------------------

elif section == "Topic & Problem Analysis":

    st.header("📌 Rural Development Topics")

    topic_data = {

"Internet":("Poor internet connectivity","Expand rural broadband"),
"Drinking Water":("Unsafe drinking water","Village water purification systems"),
"Agriculture":("Low crop productivity","Promote modern farming techniques"),
"Fertilizer":("Fertilizer shortage","Improve fertilizer supply chain"),
"Seeds":("Low quality seeds","Provide certified seeds"),
"Transport":("Poor rural transport","Improve rural transport network"),
"Markets":("Low crop prices","Improve farmer market access"),
"Storage":("Lack of storage facilities","Build rural warehouses"),
"Loans":("Limited farmer credit","Expand rural banking"),
"Subsidy":("Delay in government subsidies","Direct benefit transfer systems"),
"Drought":("Frequent drought","Rainwater harvesting"),
"Flood":("Flood damage","Flood control infrastructure"),
"Livestock":("Animal diseases","Improve veterinary services"),
"Soil":("Soil fertility decline","Promote organic farming"),
"Education":("Teacher shortage","Rural teacher recruitment"),
"Healthcare":("Lack of rural hospitals","Build rural healthcare centers"),
"Electricity":("Power cuts in villages","Improve rural power infrastructure"),
"Roads":("Poor road connectivity","Increase rural road projects"),
"Irrigation":("Water shortage for crops","Install solar irrigation pumps"),
"Crop Insurance":("Slow claim approval","Digital claim processing"),
"Cold Storage":("Food spoilage","Build cold storage units"),
"Farmer Income":("Low farm income","Introduce minimum support prices"),
"Organic Farming":("Lack of awareness","Provide farmer training"),
"Agri Technology":("Low technology adoption","Introduce smart farming tools"),
"Weather Forecasting":("Unpredictable weather","Provide weather alerts"),
"Farmer Training":("Limited knowledge","Organize agriculture workshops"),
"Rural Banking":("Limited bank access","Open rural banking branches"),
"Digital Payments":("Low digital literacy","Provide digital training"),
"Mobile Connectivity":("Weak mobile signal","Install rural towers"),
"Village Roads":("Poor road conditions","Build all-weather roads"),
"Solar Energy":("Lack of electricity","Install solar panels"),
"Women Farmers":("Limited support","Provide women farmer schemes"),
"Self Help Groups":("Lack of funding","Provide microfinance support"),
"Fisheries":("Low fish production","Improve fish farming techniques"),
"Poultry Farming":("Disease outbreaks","Improve poultry healthcare"),
"Milk Production":("Low dairy productivity","Improve dairy infrastructure"),
"Food Processing":("Low value addition","Build food processing units"),
"Village Markets":("Limited market access","Develop rural markets"),
"Farmer Cooperatives":("Poor coordination","Promote farmer cooperatives"),
"Smart Farming":("Low automation","Introduce IoT farming systems"),
"Precision Agriculture":("Inefficient farming","Use data-driven agriculture"),
"Rural Tourism":("Low tourism infrastructure","Develop rural tourism programs"),
"Village Startups":("Limited entrepreneurship","Support rural startups"),
"Micro Finance":("Limited small loans","Provide microcredit"),
"Skill Development":("Low employment skills","Provide rural skill training"),
"Crop Diversification":("Dependence on single crop","Encourage diverse crops"),
"Drip Irrigation":("Water wastage","Promote drip irrigation"),
"Rainwater Harvesting":("Water scarcity","Build rainwater harvesting systems"),
"Farm Machinery":("Limited machinery access","Subsidize farm equipment"),
"Tractor Access":("High tractor cost","Provide tractor rental schemes")
}

    topics=list(topic_data.keys())

    df_topics=pd.DataFrame({
        "Topic":topics,
        "Problem Identified":[topic_data[t][0] for t in topics],
        "Possible Solution":[topic_data[t][1] for t in topics]
    })

    # expand topics to 100 rows
    while len(df_topics) < 100:
        df_topics = pd.concat([df_topics, df_topics], ignore_index=True)

    df_topics = df_topics.head(100)

    num_topics = st.number_input(
        "Select number of rural topics",
        min_value=1,
        max_value=100,
        value=10
    )

    st.subheader(f"Top {num_topics} Rural Development Topics")

    st.dataframe(df_topics.head(num_topics), hide_index=True)




