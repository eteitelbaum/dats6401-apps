# app_pca_starter.py — Week 3: PCA explorer with a color-by selector
# Run with:  streamlit run app_pca_starter.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("PCA Explorer: 1970s–80s Cars")

# TODO: load your dataset; keep a DataFrame `num` of ONLY numeric columns

@st.cache_data
def load_and_project():
    mpg_df = sns.load_dataset("mpg").dropna() #full data
    num_df = mpg_df.select_dtypes("number") #numeric only
    X = StandardScaler().fit_transform(num_df)
    pca = PCA(2)
    pcs = pca.fit_transform(X)
    df = mpg_df.copy()
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
    df["cylinders"] = df["cylinders"].astype(str)
    return df, num_df, pca.explained_variance_ratio_

df, num_df, evr = load_and_project()
st.caption(f"PC1 {evr[0]:.0%} · PC2 {evr[1]:.0%} of variance")

# heatmap of correlations
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(num_df.corr(), cmap="vlag", center=0, square=True,
            cbar_kws={"shrink": 0.6}, ax=ax, annot=True, fmt=".2f")
ax.set_title("Correlations among the 7 numeric variables")
st.pyplot(fig)

# TODO: add a selectbox over the columns and use it as the color:
color_col = st.selectbox("Color points by", ["origin", "cylinders", "model_year", "mpg", "weight", "displacement", "horsepower", "acceleration"])

st.scatter_chart(df, x="PC1", y="PC2", color=color_col,
                 x_label=f"PC1 ({evr[0]:.0%} of variance)",
                 y_label=f"PC2 ({evr[1]:.0%} of variance)")

st.markdown("""
### Interpretation

**Correlations.** Four variables (cylinders, displacement, horsepower, and weight) are
almost interchangeable. Their pairwise correlations run from 0.84 to 0.95 whereas fuel economy (mpg)
moves strongly in the opposite direction (r ≈ −0.8 with each of them).

**PC1 (72% of variance): engine size and power vs. fuel economy.** Cylinders, displacement,
horsepower, and weight load almost equally (≈ 0.42), with mpg on the opposite side (−0.40).
High PC1 scores are big, heavy, powerful cars; low scores are light, efficient ones.
(Acceleration is measured in seconds to 60 mph, so its negative loading means big-engine cars are *quicker*.)

**PC2 (12% of variance): model year.** Model year alone dominates PC2 (loading 0.91), so the
vertical axis mostly separates older cars (1970) from newer ones (1982).

**Structure.** The cloud breaks into three clumps along PC1, which match 4-, 6-, and 8-cylinder
cars. Coloring by origin shows a lopsided pattern: Japanese and European cars overlap entirely
at the small-car end, while American cars span the whole range and are the *only* ones at the
large-engine end. Origin isn't an input to the PCA, so this pattern is a finding rather than
an artifact. Coloring by cylinders or mpg also lines up with PC1, but only because those
variables help define it.
""")
