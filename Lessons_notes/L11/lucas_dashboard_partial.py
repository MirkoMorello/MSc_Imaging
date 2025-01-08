import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import torch
from sklearn.decomposition import PCA



def plot_map(df, cur_var, soil_vars):
    # round values to 2 decimals
    df[soil_vars] = df[soil_vars].round(2)
    # create map
    fig = px.scatter_mapbox(df, lat='GPS_LAT', lon='GPS_LONG', hover_name=cur_var, hover_data=['GPS_LAT','GPS_LONG']+soil_vars,
                            color=cur_var, color_discrete_sequence=["fuchsia"], zoom=3, height=600)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # plot map
    st.plotly_chart(fig, use_container_width=True)

def get_src(df, src_prefix):
    ''' gets hyperspectral signals using cols starting with src_prefix '''
    # get x points
    src_cols = [col for col in df.columns if col.startswith(src_prefix)]
    x = np.array([float(col[len(src_prefix):]) for col in src_cols])
    # sort x points in increasing order
    pos = np.argsort(x)
    src_cols = [src_cols[cur_pos] for cur_pos in pos]
    # extract x and y values
    src_x = x[pos]
    src_y = df[src_cols].to_numpy()
    # convert to tensor
    src_x = torch.from_numpy(src_x).float()
    src_y = torch.from_numpy(src_y).float()
    # return
    return src_cols, src_x, src_y


def get_tgt(df, tgt_vars):
    ''' gets target variables using specified columns '''
    # extract variables
    tgt_vars = df[tgt_vars].to_numpy()
    # convert to torch
    tgt_vars = torch.from_numpy(tgt_vars).float()
    # return them
    return tgt_vars

@st.cache_data
def load_data(fn, df):
    # get vars
    src_cols, src_x, src_y = get_src(df, src_prefix)
    pca = PCA(n_components=0.9999)
    pca.fit(src_y)
    # return
    return src_cols, src_x, src_y, pca



if __name__ == '__main__':
    # set page width
    st.set_page_config(layout="wide")
    # define soil variables
    soil_vars = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC']
    # initialize
    uploaded_file = None
    df = pd.read_csv('../datasets/Lucas/lucas_dataset_val.csv')

    data_fn = '../datasets/Lucas/lucas_dataset_val.csv'
    src_prefix = 'spc.'

    src_cols, src_x, src_y, pca = load_data(data_fn, df)
    col1, col2 = st.columns(2)

    with col1:
        # create file uploader
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

        # create soil variable selector
        selected_soil_vars = st.multiselect("Select Soil Variables", soil_vars)

        # check if a file has been uploaded
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

    with col2:
        if selected_soil_vars:
            plot_map(df, selected_soil_vars[0], selected_soil_vars)
    
    # add other two columns underneat the first two
    col3, col4 = st.columns(2)
    with col3:
        sliders = []
        components = pca.transform(src_y)

        for i in range(components.shape[1]):
            sliders.append(st.slider(f'Component {i}', min_value=components[:,i].min(), max_value=components[:,i].max(), value=components[:,i].mean()))



    with col4:
        fig = px.line(title='Original', labels={'x':'Wavelength (nm)', 'y':'Reflectance'})
        fig.add_scatter(x=src_x, y=src_y[0], mode='lines', name='Original')
        fig.add_scatter(x=src_x, y=pca.inverse_transform(np.array(sliders).reshape(1,-1))[0], mode='lines', name='PCA Inverse Transform')
        st.plotly_chart(fig, use_container_width=True)
        
        for i in range(len(soil_vars)):
            st.metric(label=soil_vars[i], value=df[soil_vars[i]].mean(), delta=df[soil_vars[i]].std())
