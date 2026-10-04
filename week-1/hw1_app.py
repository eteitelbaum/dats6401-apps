# app_starter.py — Week 1: your first Streamlit app
# Run with:  streamlit run app_starter.py
import streamlit as st
import plotly.express as px

st.title("Gapminder App")            

df = px.data.gapminder()                

# Group by country
by_country = (
    df.groupby(["country", "continent"], as_index=False)
        .agg(
            mean_life=("lifeExp", "mean"),
            mean_gdp=("gdpPercap", "mean")
        )
)

st.caption(f"{len(df)} rows")
st.dataframe(df.head(20))

# interactive scatter plot
fig = px.scatter(
    by_country, x="mean_gdp", y="mean_life",
    log_x=True,
    labels={
        "mean_gdp": "GDP per Capita ($)",
        "mean_life": "Life Expectancy (years)"
    },
    title="Wealth is Positively Correlated with Life Expectancy"
)
fig.update_xaxes(
    tickvals=[500, 1000, 2000, 5000, 10000, 20000, 50000]
)
st.plotly_chart(fig)
st.info(
    "Plot displays the relationship between average per capita GDP and life expectancy "
    " for the period 1952-2007. Data are from the Gapminder Dataset."    
    )