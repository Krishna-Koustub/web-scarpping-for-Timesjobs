import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

def musixmatch_scatplot(df,df_embed,n_components=2):
    
    """Comprehensive function that adds components to a scatter plot. Requires musixmatch_PCA function """
    
    df = musixmatch_PCA(df,df_embed,n_components) #gets PCA
             
    fig = go.Figure() #instantiate Ploty.go object

    genres = df.genre.unique() #get unique genres     
    
    for i in range(len(genres)): #loop through genres

        df_mask = df[df.genre == genres[i]] #subset on genre
        
        df_mask['artist_song'] = df_mask['artist']+' // '+df_mask['song'] #new labels for traces
             
        if n_components == 2:
            
            #add traces for 2d - several formatting options for hover text, and marker size
            fig.add_trace(go.Scatter(
                          x=df_mask['pca_x'],
                          y=df_mask['pca_y'],
                          name=genres[i],
                          text=df_mask['artist_song'],
                          mode='markers',hoverinfo='text',
                          marker={'size':df_mask.lyric_count_norm}))
        
        else:
            #add traces for 3d - several formatting options for hover text, and marker size
            fig.add_trace(go.Scatter3d(
                          x=df_mask['pca_x'],
                          y=df_mask['pca_y'],
                          z=df_mask['pca_z'],
                          name=genres[i],
                          text=df_mask['artist_song'],
                          mode='markers',hoverinfo='text',
                          marker={'size':df_mask.lyric_count_norm}))
    # axis parameters
    axis_x_param=dict(showline=True, 
                      zeroline=True,
                      showgrid=True,
                      showticklabels=True,
                      title='Principal Component 1')
    # axis parameters
    axis_y_param=dict(showline=True, 
                      zeroline=True,
                      showgrid=True,
                      showticklabels=True,
                      title='Principal Component 2')
    
    # legend parameters
    legend_param= dict(bgcolor=None,
                       bordercolor = None,
                       borderwidth = None,
                       font = dict(family='Open Sans',size=15,color=None),
                       orientation='h',
                       itemsizing='constant',
                       title=dict(text='Genres (clickable!)',
                                  font=dict(family='Open Sans',size=20,color=None),
                                  side='top'),)
    # margin parameters
    margin_param=dict(l=40,r=40,b=85,t=200,pad=0)
    
    # title parameters
    title_param = dict(text='<b>Similarities and differences in song lyrics by genre</b>\
    <br>Universal sentence encodings and dimensionality reduction - trace size represents lyric count.', 
                        font=dict(size=20))
    
    #update layout
    fig.update_layout(legend= legend_param,
                      title=title_param,
                      width=1000,
                      height=1000,
                      autosize=False,
                      showlegend=True,
                      xaxis=axis_x_param,
                      yaxis=axis_y_param,
                      margin=margin_param,)
    
        


    fig.show()    
    return fig