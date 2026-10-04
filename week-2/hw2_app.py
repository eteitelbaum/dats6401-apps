# app_compare_starter.py — Week 2: encodings side by side
# Run with:  streamlit run app_compare_starter.py
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.title("One Question, Two Encodings")
st.caption("Which continent had the largest population in 1952?")

df = px.data.gapminder()      # TODO: your dataset

#filter for 1952
df1952 = df[df["year"] == 1952]
df1952.head()  

# Group by country
pop_by_region_1952 = (
    df1952.groupby(["continent"], as_index=False)
        .agg(
            mean_life=("lifeExp", "mean"),
            mean_gdp=("gdpPercap", "mean"),
            pop_sum=("pop", "sum")
        )
)

# put population in millions
pop_by_region_1952["pop_millions"] = pop_by_region_1952["pop_sum"] / 1e6

# sort the data by population
pop_by_region_1952 = pop_by_region_1952.sort_values("pop_sum", ascending=False)

left, right = st.columns(2, vertical_alignment="center")

with left:
    st.subheader("Encoding A")
    fig, ax = plt.subplots(figsize=(5,4))
    
    # TODO: your STRONG encoding
    sns.barplot(data=pop_by_region_1952, x="continent", y="pop_millions", ax=ax)
    ax.set(xlabel="Continent", ylabel="Population (millions)",title="Population by Region, 1952")
    st.pyplot(fig)
    st.caption("Channel: Position on a common scale · ranking position: 1")

with right:
    st.subheader("Encoding B")
    fig, ax = plt.subplots(figsize=(5,4))

    # TODO: your second encoding (the weak one is more instructive!)
    ax.pie(pop_by_region_1952["pop_millions"], labels=pop_by_region_1952["continent"])
    ax.set_title("Population by Region, 1952")    
    st.pyplot(fig)
    st.caption("Channel: Angle/area · ranking position: 4/5")
