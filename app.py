import streamlit as st
import pandas as pd
from model import UnsupervisedFuzzyClusterer, SemiSupervisedFuzzyClusterer

from utils import prepare_iris
from metrics import fuzzy_partition_coefficient, partition_entropy, calinski_harabasz_index, rand_index, adjusted_rand_index, jaccard_coefficient
from utils import plot
from utils import show_centroids
from utils import unsupervisedPrepare_iris, true_labels


st.set_page_config(
    page_title='Fuzzy clustering',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('Fuzzy clustering')
c1, c2 = st.columns([0.4, 0.6], gap='large')

csv_file = c1.file_uploader('Upload data', type='csv')

c1.header('Parameters')
m = c1.number_input('m', min_value=1.0, value=2.0)
C = c1.number_input('C', min_value=2, value=3)
eps = c1.number_input('eps', min_value=1e-10, step=1e-10, format='%e')
type = c1.radio('Type', ['Unsupervised', 'Semi-supervised'])

clicked = c1.button('Continue')

if csv_file is not None and clicked:
    
    if type == 'Unsupervised':
        if csv_file.name == 'iris.csv':
            data = unsupervisedPrepare_iris(csv_file, C)
            y = data
        else:
            data = pd.read_csv(csv_file)
            y = data.to_numpy(float)
        model = UnsupervisedFuzzyClusterer(len(data), C, data.shape[1], m=m)
        losses = model.fit(y, eps)
        U = model.U
        v = model.v
    else:
        x, U_bar = prepare_iris(csv_file, C)
        y = x
        model = SemiSupervisedFuzzyClusterer(
            N=len(x),
            C=C,
            D=x.shape[-1],
            m=m
        )
        losses = model.fit(x, U_bar, eps)
        U = model.U.T
        v = model.v
        
    
    
    c2.header('Centeroids')
    show_centroids(c2, v)
    
    if y.shape[-1] == 2:
        c2.header('2D plot')
        c2.pyplot(plot(y, U, v))
    
    c2.header('Metrics')
    c2.write(f'Loss (Jm): {losses[0]} -> {losses[-1]}')
    c2.write(f'Partition Coefficient (Fc): {fuzzy_partition_coefficient(U)}')
    c2.write(f'1 - Fc: {1 - fuzzy_partition_coefficient(U)}')
    c2.write(f'Entropy (Hc): {partition_entropy(U)}')
    c2.write(f'Calinski-Harabasz (VRC): {calinski_harabasz_index(y, U)}')
    
    c2.write(f'Rand Index (RI): {rand_index(true_labels(), U)}')
    c2.write(f'Adjusted Rand Index (ARI): {adjusted_rand_index(true_labels(), U)}')
    c2.write(f'Jaccard Coefficient (Jc): {jaccard_coefficient(true_labels(), U)}')