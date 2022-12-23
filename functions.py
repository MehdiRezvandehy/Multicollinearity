# Required function for running this assignment
# Written by Mehdi Rezvandehy


import pandas as pd
import numpy as np
import time
import matplotlib 
import pylab as plt
from scipy.stats import zscore
from matplotlib import gridspec
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import PercentFormatter
from sklearn import tree
from sklearn.datasets import make_swiss_roll
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import Pool, CatBoostRegressor
from catboost import Pool, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import ppscore as pps


##########################################################################################################

class Correlation_plot:
    def corr_mat(df: pd.DataFrame, title: str, corr_val_font: float=False, y_l: list=1.2,axt: plt.Axes=None,
                titlefontsize: int=10, xyfontsize: int=6, xy_title: list=[-22,1.2],
                vlim=[-0.8,0.8]) -> [float]:
        
        """Plot correlation matrix between features"""
        ax = axt or plt.axes()
        colmn=list(df.columns)
        corr=df.corr().values
        corr_array=[]
        for i in range(len(colmn)):
            for j in range(len(colmn)):
                c=corr[j,i]
                if (corr_val_font):
                        ax.text(j, i, str(round(c,1)), va='center', ha='center',fontsize=corr_val_font)
                if i>j:
                    corr_array.append(c)

        im =ax.matshow(corr, cmap='jet', interpolation='nearest',vmin=vlim[0], vmax=vlim[1])

        cbar =plt.colorbar(im,shrink=0.5,label='Correlation Coefficient')
        cbar.ax.tick_params(labelsize=10) 
        
        ax.set_xticks(np.arange(len(corr)))
        ax.set_xticklabels(colmn,fontsize=xyfontsize, rotation=90)
        ax.set_yticks(np.arange(len(corr)))
        ax.set_yticklabels(colmn,fontsize=xyfontsize)
        ax.grid(color='k', linestyle='-', linewidth=0.025)
        plt.text(xy_title[0],xy_title[1],title, 
                 fontsize=titlefontsize,bbox=dict(facecolor='white', alpha=0.2))
        return corr_array
        plt.show()
        
        
    #########################  
    
    def corr_bar(corr: list, clmns: str,title: str, select: bool= False
                ,yfontsize: float=4.6, xlim: list=[-0.5,0.5], ymax_vert_lin: float= False) -> None:
        
        """Plot correlation bar with target"""
        
        r_ = pd.DataFrame( { 'coef': corr, 'positive': corr>=0  }, index = clmns )
        r_ = r_.sort_values(by=['coef'])
        if (select):
            selected_features=abs(r_['coef'])[:select].index
            r_=r_[r_.index.isin(selected_features)]
    
        r_['coef'].plot(kind='barh',edgecolor='black',linewidth=0.8
                        , color=r_['positive'].map({True: 'r', False: 'b'}))
        plt.xlabel('Correlation Coefficient',fontsize=6)
        if (ymax_vert_lin): plt.vlines(x=0,ymin=-0.5, ymax=ymax_vert_lin, color = 'k',linewidth=0.5)
        plt.yticks(np.arange(len(r_.index)), r_.index,rotation=0,fontsize=yfontsize,x=0.01)
        plt.title(title)
        plt.xlim((xlim[0], xlim[1])) 
        ax1 = plt.gca()
        ax1.xaxis.grid(color='k', linestyle='-', linewidth=0.1)
        ax1.yaxis.grid(color='k', linestyle='-', linewidth=0.1)
        plt.show()   
        
############################################################################# 

class prfrmnce_plot(object):
    """Plot performance of features to predict a target"""
    def __init__(self,importance: list, title: str, ylabel: str,clmns: str,
                titlefontsize: int=10, xfontsize: int=5, yfontsize: int=8) -> None:
        self.importance    = importance
        self.title         = title 
        self.ylabel        = ylabel  
        self.clmns         = clmns  
        self.titlefontsize = titlefontsize 
        self.xfontsize     = xfontsize 
        self.yfontsize     = yfontsize
        
    #########################    
    
    def bargraph(self, select: bool= False, fontsizelable: bool= False, xshift: float=-0.1, nsim: int=False
                 ,yshift: float=0.01,perent: bool=False, xlim: list=False,axt=None,
                 ylim: list=False, y_rot: int=0, graph_float: bool=True) -> pd.DataFrame():
        ax1 = axt or plt.axes()
        if not nsim:
            # Make all negative coefficients to positive
            sort_score=sorted(zip(abs(self.importance),self.clmns), reverse=True)
            Clmns_sort=[sort_score[i][1] for i in range(len(self.clmns))]
            sort_score=[sort_score[i][0] for i in range(len(self.clmns))]
        else:
            importance_agg=[]
            importance_std=[]
            for iclmn in range(len(self.clmns)):
                tmp=[]
                for isim in range(nsim):
                    tmp.append(abs(self.importance[isim][iclmn]))
                importance_agg.append(np.mean(tmp))
                importance_std.append(np.std(tmp))
                
            # Make all negative coefficients to positive
            sort_score=sorted(zip(importance_agg,self.clmns), reverse=True)
            Clmns_sort=[sort_score[i][1] for i in range(len(self.clmns))]
            sort_score=[sort_score[i][0] for i in range(len(self.clmns))]                
            

        index1 = np.arange(len(self.clmns))
        # select the most important features
        if (select):
            Clmns_sort=Clmns_sort[:select]
            sort_score=sort_score[:select]
        ax1.bar(Clmns_sort, sort_score, width=0.6, align='center', alpha=1, edgecolor='k', capsize=4,color='b')
        plt.title(self.title,fontsize=self.titlefontsize)
        ax1.set_ylabel(self.ylabel,fontsize=self.yfontsize)
        ax1.set_xticks(np.arange(len(Clmns_sort)))
        
        ax1.set_xticklabels(Clmns_sort,fontsize=self.xfontsize, rotation=90,y=0.02)   
        if (perent): plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        ax1.xaxis.grid(color='k', linestyle='--', linewidth=0.2) 
        if (xlim): plt.xlim(xlim)
        if (ylim): plt.ylim(ylim)
        if (fontsizelable):
            for ii in range(len(sort_score)):
                if (perent):
                    plt.text(xshift+ii, sort_score[ii]+yshift,f'{"{0:.1f}".format(sort_score[ii]*100)}%',
                    fontsize=fontsizelable,rotation=y_rot,color='k')     
                else:
                    if graph_float:
                        plt.text(xshift+ii, sort_score[ii]+yshift,f'{"{0:.3f}".format(sort_score[ii])}',
                        fontsize=fontsizelable,rotation=y_rot,color='k') 
                    else:
                        plt.text(xshift+ii, sort_score[ii]+yshift,f'{"{0:.0f}".format(sort_score[ii])}',
                            fontsize=fontsizelable,rotation=y_rot,color='k')                             
                    
        
        dic_Clmns={}
        for i in range(len(Clmns_sort)):
            dic_Clmns[Clmns_sort[i]]=sort_score[i]
            
        return  pd.DataFrame(dic_Clmns.items(), columns=['Features', 'Scores'])  
        plt.show()   
        
    #########################    
    
    def Conf_Matrix(y_train: [float],y_train_pred:[float],perfect: int= 0,axt=None,plot: bool =True,
                   title: bool =False,t_fontsize: float =8.5,t_y: float=1.2,x_fontsize: float=6.5,
                   y_fontsize: float=6.5,trshld: float=0.5) -> [float]:
        
        '''Plot confusion matrix'''
        
        if (y_train_pred.shape[1]==2):
            y_train_pred=[0 if y_train_pred[i][0]>trshld else 1 for i in range(len(y_train_pred))]
        elif (y_train_pred.shape[1]==1):
            y_train_pred=[1 if y_train_pred[i][0]>trshld else 0 for i in range(len(y_train_pred))] 
        else:    
            y_train_pred=[1 if i>trshld else 0 for i in y_train_pred]       
        conf_mx=confusion_matrix(y_train,y_train_pred)
        acr=accuracy_score(y_train,y_train_pred)
        conf_mx =confusion_matrix(y_train,y_train_pred)
        prec=precision_score(y_train,y_train_pred) # == TP/(TP+FP) 
        reca=recall_score(y_train,y_train_pred) # == TP/(TP+FN) ) 
        TN=conf_mx[0][0] ; FP=conf_mx[0][1]
        spec= TN/(TN+FP)        
        if(plot):
            ax1 = axt or plt.axes()
            
            if (perfect==1): y_train_pred=y_train
            
            x=['Predicted \n Negative', 'Predicted \n Positive']; y=['Actual \n Negative', 'Actual \n Positive']
            ii=0 
            im =ax1.matshow(conf_mx, cmap='jet', interpolation='nearest') 
            for (i, j), z in np.ndenumerate(conf_mx): 
                if(ii==0): al='TN= '
                if(ii==1): al='FP= '
                if(ii==2): al='FN= '
                if(ii==3): al='TP= '          
                ax1.text(j, i, al+'{:0.0f}'.format(z), color='w', ha='center', va='center', fontweight='bold',fontsize=6.5)
                ii=ii+1
         
            txt='$ Accuracy\,\,\,$=%.2f\n$Sensitivity$=%.2f\n$Precision\,\,\,\,$=%.2f\n$Specificity$=%.2f'
            anchored_text = AnchoredText(txt %(acr,reca,prec,spec), loc=10, borderpad=0)
            ax1.add_artist(anchored_text)    
            
            ax1.set_xticks(np.arange(len(x)))
            ax1.set_xticklabels(x,fontsize=x_fontsize,y=0.97, rotation='horizontal')
            ax1.set_yticks(np.arange(len(y)))
            ax1.set_yticklabels(y,fontsize=y_fontsize,x=0.035, rotation='horizontal') 
            
            cbar =plt.colorbar(im,shrink=0.3,
                               label='Low                              High',orientation='vertical')   
            cbar.set_ticks([])
            plt.title(title,fontsize=t_fontsize,y=t_y)
        return acr, prec, reca, spec    
        
    #########################    
    
    def AUC(prediction: [float],y_train: [float], n_algorithm: int
           ,label:[str],title: str='Receiver Operating Characteristic (ROC)'
           ,linewidth=2) -> None:
        
        '''Plot Receiver Operating Characteristic (ROC) for predictors'''
        
        color=['b','r','g','y','b']
        for i in range(n_algorithm):
            fpr, tpr, thresold = roc_curve(y_train, prediction_prob[i][:,1])
            roc_auc = auc(fpr, tpr)
            if (i==0):
                tmp_linewidth=4
                cm='k--'
            else:
                tmp_linewidth=linewidth
                cm= f'{color[i]}-'
                
            plt.plot(fpr, tpr,cm, linewidth=tmp_linewidth,
                     label=label[i]+' (AUC =' + r"$\bf{" + str(np.round(roc_auc,3)) + "}$"+')')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1-Specificity) FP/(FP+TN)',fontsize=12)
        plt.ylabel('True Positive Rate (Sensistivity) TP/(TP+FN)',fontsize=12)
        plt.title(title,fontsize=15)
        plt.grid(linewidth='0.25')
        plt.legend(loc="lower right",fontsize=11)
        plt.show()    

########################################################

class EDA_plot:
    
    def histplt (val: list,bins: int,title: str,xlabl: str,ylabl: str,xlimt: list,
                 ylimt: list=False, loc: int =1,legend: int=1,axt=None,days: int=False,
                 class_: int=False,scale: int=1,x_tick: list=False,
                 nsplit: int=1,font: int=5,color: str='b') -> None :
        
        """ Make histogram of data"""
        
        ax1 = axt or plt.axes()
        font = {'size'   : font }
        plt.rc('font', **font) 
        
        val=val[~np.isnan(val)]
        val=np.array(val)
        plt.hist(val, bins=bins, weights=np.ones(len(val)) / len(val),ec='black',color=color)
        n=len(val[~np.isnan(val)])
        Mean=np.nanmean(val)
        Median=np.nanmedian(val)
        SD=np.sqrt(np.nanvar(val))
        Max=np.nanmax(val)
        Min=np.nanmin(val)
    
        
        txt='n=%.0f\nMean=%0.2f\nMedian=%0.1f\nσ=%0.1f\nMax=%0.1f\nMin=%0.1f'       
        anchored_text = AnchoredText(txt %(n,Mean,Median,SD,Max,Min), borderpad=0, 
                                     loc=loc,prop={ 'size': font['size']*scale})    
        if(legend==1): ax1.add_artist(anchored_text)
        if (scale): plt.title(title,fontsize=font['size']*(scale+0.15))
        else:       plt.title(title)
        plt.xlabel(xlabl,fontsize=font['size']) 
        ax1.set_ylabel('Frequency',fontsize=font['size'])
        if (scale): ax1.set_xlabel(xlabl,fontsize=font['size']*scale)
        else:       ax1.set_xlabel(xlabl)
    
        try:
            xlabl
        except NameError:
            pass    
        else:
            if (scale): plt.xlabel(xlabl,fontsize=font['size']*scale) 
            else:        plt.xlabel(xlabl)   
            
        try:
            ylabl
        except NameError:
            pass      
        else:
            if (scale): plt.ylabel(ylabl,fontsize=font['size']*scale)  
            else:         plt.ylabel(ylabl)  
            
        if (class_==True): plt.xticks([0,1])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        ax1.grid(linewidth='0.1')
        try:
            xlimt
        except NameError:
            pass  
        else:
            plt.xlim(xlimt) 
            
        try:
            ylimt
        except NameError:
            pass  
        else:
            plt.ylim(ylimt)  
            
        if x_tick: plt.xticks(x_tick,fontsize=font['size']*scale)    
        plt.yticks(fontsize=font['size']*scale)               
    
    ######################################################################### 
            
    def KDE(xs: list,data_var: list,nvar: int,clmn: [str],color: [str],xlabel: str='DE Length',
            title: str='Title',ylabel: str='Percentage',LAMBA: float =0.3,linewidth: float=2.5,
            loc: int=0,axt=None,xlim: list=(0,40),ylim: list=(0,0.1),x_ftze: float =13,
            y_ftze: float=13,tit_ftze: float=13,leg_ftze: float=9) -> None :
        
        """
        Kernel Density Estimation (Smooth Histogram)
         
        """
        ax1 = axt or plt.axes()
        var_m=[]
        var_med=[]
        var_s=[]
        var_n=[]
        s1=[]
        data_var_=np.array([[None]*nvar]*len(xs), dtype=float)
        # Loop over variables
        for i in range (nvar):
            data = data_var[i]
            var_m.append(data.mean().round(2))
            var_med.append(np.median(data).round(2))
            var_s.append(np.sqrt(data.var()).round(1))
            var_n.append(len(data))
            density = gaussian_kde(data)
            density.set_bandwidth(LAMBA)
            density_=density(xs)/sum(density(xs))
            data_var_[:,i]=density_
            linestyle='solid'
            plt.plot(xs,density_,color=color[i],linestyle=linestyle, linewidth=linewidth)
            
        #############
        
        data_var_tf=np.array([[False]*nvar]*len(data_var_))
        for j in range(len(data_var_)):
            data_tf_t=[]
            for i in range (nvar):
                if (data_var_[j,i]==max(data_var_[j,:])):
                    data_var_tf[j,i]=True     
        #############            
        for i in range (nvar):
            plt.fill_between(np.array(xs),np.array(data_var_[:,i]),where=np.array(data_var_tf[:,i]),
                             color=color[i],alpha=0.9,label=clmn[i]+': n='+str(var_n[i])+
                             ', mean= '+str(var_m[i])+', median= '+str(var_med[i])+
                             ', '+r"$\sigma$="+str(var_s[i]))
        
        plt.xlabel(xlabel,fontsize=x_ftze, labelpad=6)
        plt.ylabel(ylabel,fontsize=y_ftze)
        plt.title(title,fontsize=tit_ftze)
        plt.legend(loc=loc,fontsize=leg_ftze,markerscale=1.2)
        
        ax1.grid(linewidth='0.2')
        plt.xlim(xlim) 
        plt.ylim(ylim) 

    ######################################################################### 
            
    def CDF_plot(data_var: list,nvar: int,label:str,colors:str,title:str,xlabel:str,
                 ylabel:str='Cumulative Probability', bins: int =1000,xlim: list=(0,100),
                 ylim: list=(0,0.01),linewidth: float =2.5,loc: int=0,axt=None,
                 x_ftze: float=12,y_ftze: float=12,tit_ftze: float=12,leg_ftze: float=9) -> None:
        
        """
        Cumulative Distribution Function
         
        """
        ax1 = axt or plt.axes() 
        
        # Loop over variables
        for i in range (nvar):
            data = data_var[i]
            var_mean=np.nanmean(data).round(2)
            var_median=np.nanmedian(data).round(2)
            var_s=np.std(data).round(1)
            var_n=len(data)
            val_=np.array(data)
            counts, bin_edges = np.histogram(val_[~np.isnan(val_)], bins=bins,density=True)
            cdf = np.cumsum(counts)
            tmp=max(cdf)
            cdf=cdf/float(tmp)
            label_=f'{label[i]} : n={var_n}, mean= {var_mean}, median= {var_median}, $\sigma$={var_s}'
            plt.plot(bin_edges[1:], cdf,color=colors[i], linewidth=linewidth,
                    label=label_)
         
        plt.xlabel(xlabel,fontsize=x_ftze, labelpad=6)
        plt.ylabel(ylabel,fontsize=y_ftze)
        plt.title(title,fontsize=tit_ftze)
        plt.legend(loc=loc,fontsize=leg_ftze,markerscale=1.2)
        
        ax1.grid(linewidth='0.2')
        plt.xlim(xlim) 
        plt.ylim(ylim)         

    ######################################################################### 
            
    def CrossPlot (x:list,y:list,title:str,xlabl:str,ylabl:str,loc:int,
                   xlimt:list,ylimt:list,axt=None,scale: float=0.8,alpha: float=0.6,
                   markersize: float=6,marker: str='ro') -> None:
        #
        ax1 = axt or plt.axes()
        x=np.array(x)
        y=np.array(y)    
        no_nan=np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
        Mean_x=np.mean(x)
        SD_x=np.sqrt(np.var(x)) 
        #
        n_x=len(x)
        n_y=len(y)
        Mean_y=np.mean(y)
        SD_y=np.sqrt(np.var(y)) 
        corr=np.corrcoef(x[no_nan],y[no_nan])
        n_=len(no_nan)
        #txt=r'$\rho_{x,y}=$%.2f'+'\n $n=$%.0f '
        #anchored_text = AnchoredText(txt %(corr[1,0], n_),borderpad=0, loc=loc,
        #                         prop={ 'size': font['size']*0.95, 'fontweight': 'bold'})  
        
        
        txt=r'$\rho_{x,y}}$=%.2f'+'\n $n$=%.0f \n $\mu_{x}$=%.0f \n $\sigma_{x}$=%.0f \n '
        txt+=' $\mu_{y}$=%.0f \n $\sigma_{y}$=%.0f'
        anchored_text = AnchoredText(txt %(corr[1,0], n_x,Mean_x,SD_x,Mean_y,SD_y), loc=4,
                                prop={ 'size': font['size']*1.1, 'fontweight': 'bold'})    
            
        ax1.add_artist(anchored_text)
        Lfunc1=np.polyfit(x,y,1)
        vEst=Lfunc1[0]*x+Lfunc1[1]    
        try:
            title
        except NameError:
            pass  # do nothing! 
        else:
            plt.title(title,fontsize=font['size']*(scale))   
    #
        try:
            xlabl
        except NameError:
            pass  # do nothing! 
        else:
            plt.xlabel(xlabl,fontsize=font['size']*scale)            
    #
        try:
            ylabl
        except NameError:
            pass  # do nothing! 
        else:
            plt.ylabel(ylabl,fontsize=font['size']*scale)        
            
        try:
            xlimt
        except NameError:
            pass  # do nothing! 
        else:
            plt.xlim(xlimt)   
    #        
        try:
            ylimt
        except NameError:
            pass  # do nothing! 
        else:
            plt.ylim(ylimt)   
          
        plt.plot(x,y,marker,markersize=markersize,alpha=alpha)   
        ax1.plot(x, vEst,'k-',linewidth=2)   
        ax1.grid(linewidth='0.1') 
        plt.xticks(fontsize=font['size']*0.85)    
        plt.yticks(fontsize=font['size']*0.85)            
        
            
####################################################################    

def CrossPlot (x,y,title,xlabl,ylabl,loc,xlimt,ylimt,font: dict,axt=None,scale=0.8,alpha=0.4,markersize=6,marker='ro'):
    #
    ax1 = axt or plt.axes()
    x=np.array(x)
    y=np.array(y)    
    no_nan=np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
    Mean_x=np.mean(x)
    SD_x=np.sqrt(np.var(x)) 
    #
    n_y=len(y)
    Mean_y=np.mean(y)
    SD_y=np.sqrt(np.var(y)) 
    corr=np.corrcoef(x[no_nan],y[no_nan])
    n_=len(no_nan)
    txt=r'$\rho_{x,y}=$%.2f'
    anchored_text = AnchoredText(txt %(corr[1,0]),borderpad=0, loc=loc,
                             prop={ 'size': font['size']*0.85, 'fontweight': 'bold'})     
        
    ax1.add_artist(anchored_text)
    #Lfunc1=np.polyfit(x,y,1)
    #vEst=Lfunc1[0]*x+Lfunc1[1]    
    try:
        title
    except NameError:
        pass  # do nothing! 
    else:
        plt.title(title,fontsize=font['size']*(scale))   
#
    try:
        xlabl
    except NameError:
        pass  # do nothing! 
    else:
        plt.xlabel(xlabl,fontsize=font['size']*scale)            
#
    try:
        ylabl
    except NameError:
        pass  # do nothing! 
    else:
        plt.ylabel(ylabl,fontsize=font['size']*scale)        
        
    try:
        xlimt
    except NameError:
        pass  # do nothing! 
    else:
        plt.xlim(xlimt)   
#        
    try:
        ylimt
    except NameError:
        pass  # do nothing! 
    else:
        plt.ylim(ylimt)   
      
    
    plt.plot(x,y,marker,markersize=markersize,alpha=alpha)   
    #ax1.plot(x, vEst,'b-',linewidth=3)   
    ax1.grid(linewidth='0.1') 
    plt.xticks(fontsize=font['size']*0.85)    
    plt.yticks(fontsize=font['size']*0.85)        
    
####################################################################

    
def histplt (val,bins,title,xlabl,ylabl,xlimt,ylimt,font: dict,tot=10000,axt=None,scale=0.55,missin_rep=True):
    #
    ax1 = axt or plt.axes()
    val=val[~np.isnan(val)]
    
    val=val[~np.isnan(val)]
    
    val=np.array(val)
    plt.hist(val, bins=bins, weights=np.ones(len(val)) / len(val),ec='black')
    n=len(val[~np.isnan(val)])    
      

    Mean=np.mean(val)
    SD=np.sqrt(np.var(val)) 
    Max=np.amax(val)
    Min=np.amin(val)
    if missin_rep:
        txt='$n=$%.0f \n'+'$Missing=$%.0f'+'%%'+', $μ=$%0.0f\n$σ=$%0.0f'
        per=((10000.0-n)/10000.0)*100
        anchored_text = AnchoredText(txt %(n,per,Mean,SD), loc=1,prop={ 'size': font['size']*0.85, 'fontweight': 'bold'})   
    else:
        txt='$n=$%.0f \n'+'$μ=$%0.0f, $σ=$%0.0f'
        anchored_text = AnchoredText(txt %(n,Mean,SD), loc=1,prop={ 'size': font['size']*0.85, 'fontweight': 'bold'})         
        
    ax1.add_artist(anchored_text)
    try:
        title
    except NameError:
        plt.title('Histogram',fontsize=font['size']*(scale),rotation=90)    
    else:
        plt.title(title,fontsize=font['size']*(scale))   
#
    try:
        xlabl
    except NameError:
        pass  # do nothing! 
        plt.xlabel('X',fontsize=font['size']*scale)      
    else:
        plt.xlabel(xlabl,fontsize=font['size']*scale)    
        
    try:
        ylabl
    except NameError:
        pass  # do nothing!     
    else:
        plt.ylabel(ylabl,fontsize=font['size']*scale)          
        
#        
    try:
        xlimt
    except NameError:
        pass  # do nothing! 
    else:
        plt.xlim(xlimt)   
#        
    try:
        ylimt
    except NameError:
        pass  # do nothing! 
    else:
        plt.ylim(ylimt)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))    
    ax1.grid(linewidth='0.1')    
    plt.xticks(fontsize=font['size']*0.85)    
    plt.yticks(fontsize=font['size']*0.85)     
    
####################################################################    

    
def cplotmatrix(df,font: dict,alpha=0.8,marker='ro',missin_rep=True):
    Samp=df.values
    Samp_=df.dropna().values
    columns=df.columns
    ir=0;nvar=len(columns);title='';ik=0
    xlabl=None;ylabl=None
    #nloc=[2,4,1,2,4,1]
    for j in range (nvar):
        for i in range(nvar):
            title='';xlabl=None;ylabl=None
            ir+=1
            if (i==j):
                ax1 = plt.subplot(nvar,nvar,ir)
                x_=Samp[:,i]
                xlimt=(np.nanmin(x_),np.nanmax(x_))
                if(i==0 and j==0): title=columns[i];ylabl=columns[i];
                else: title=columns[i];xlabl=None;ylabl=None      
                histplt (x_,15,title=title,
                     xlabl=xlabl,ylabl=ylabl,xlimt=xlimt,ylimt=(0,0.4),axt=ax1,scale=1.25,missin_rep=missin_rep,font=font)
            elif(i<=j):
                ax1 = plt.subplot(nvar,nvar,ir)
                title=''
                xlabl=None;ylabl=None
                x_=Samp[:,i] ; y_=Samp[:,j]
                if(j==0): title=columns[i]
                elif(i==0): ylabl=columns[j]   
                CrossPlot(x_,y_,title=title,
                          xlabl=xlabl,ylabl=ylabl,loc=3,xlimt=(-3.5,np.nanmax(x_)),
                          ylimt=(-3.5,np.nanmax(y_)),axt=ax1,scale=1.25,alpha=alpha,marker=marker,font=font)
                ik+=1
    plt.tight_layout(pad=0.5)
    plt.show()  
    
####################################################################  

def quadrac(noise=1.2,nsim=1000,time_rand1=0.3,seed=56,a1=30, b1=3, a2=35, b2=4, swiss_roll=True):
    
    # Make a quadratic function
    np.random.seed(seed)
    #X = time_rand1*np.random.normal(0, 1, nsim).reshape(nsim,1)         # Generate random numbers from a Uniform distribution 
    X= 8 * np.random.rand(nsim, 1)-3
    
    y_no_L =-0.9* X**2 +1.5*X+ 5+time_rand1*np.random.randn(nsim, 1)   # Add random noise from a Uniform distribution          
    
    
    # Make a positive correlation
    np.random.seed(seed+10)    
    rand=a1*np.random.randn(nsim, 1)
    y_p=[-1 + b1 * y_no_L[i] + rand[i][0] for i in range(len(rand))]
    y_p=[y_p[i][0] for i in range(len(y_p))]
    
    # Make a negative correlation
    np.random.seed(seed+20)
    y_n=np.random.normal(0, 1, nsim)  
    
    # Convert the correlated distributions to Pandas dataframe
    columns=['Var ' +str(i+1) for i in range(3)]+['Target']
    
    y_no_L=[y_no_L[i][0] for i in range(len(y_no_L))]
    X=[X[i][0] for i in range(len(X))]
    df=pd.DataFrame({columns[0]:np.array(y_p),columns[1]:np.array(X),
                  columns[2]:np.array(y_n),columns[3]:np.array(y_no_L)},columns=columns) 
    return df     

def swiss_roll(noise=1.2,nsim=1000,seed=32,time_rand1=0.3,a1=30, b1=3, a2=35, b2=4, swiss_roll=True):
    
    # Make a swiss_roll and linear correlations
    if (swiss_roll):
        val, _ = make_swiss_roll(n_samples=nsim, noise=noise, random_state=seed)
        X=val[:, 0]
        y_no_L=val[:, 2]   
        
    # Make a quadratic function
    np.random.seed(seed)
    #X = time_rand1*np.random.normal(0, 1, nsim).reshape(nsim,1)         # Generate random numbers from a Uniform distribution 
    X= 8 * np.random.rand(nsim, 1)-3
    y_q =-0.9* X**2 +1.5*X+ 5+time_rand1*np.random.randn(nsim, 1)   # Add random noise from a Uniform distribution           

    # Make a positive correlation
    np.random.seed(seed+30)    
    rand=a1*np.random.randn(nsim, 1)
    y_p=[-1 + b1 * X[i] + rand[i][0] for i in range(len(rand))]

    # Make a negative correlation
    np.random.seed(seed+20)
    rand=a2*np.random.randn(nsim, 1)
    y_n=[20 - b2 * X[i] + rand[i][0] for i in range(len(rand))]    
    
    # Convert the correlated distributions to Pandas dataframe
    columns=['Var ' +str(i+1) for i in range(4)]+['Target']
    df=pd.DataFrame({columns[0]:np.array(y_no_L),columns[1]:np.array(y_p),
                  columns[2]:np.array(y_n),columns[3]:np.array(y_q)  
                  ,columns[4]:np.array(X)},columns=columns) 
    return df 

def quadrac_swiss_roll(noise=2,nsim=100000,time_rand1=1,seed=42,a1=20,b1=8,a2=15,b2=8,swiss_roll=True):
    # Make a swiss_roll and linear correlations
    df=pd.DataFrame()
    if (swiss_roll):
        val, _ = make_swiss_roll(n_samples=nsim, noise=noise, random_state=seed)
        X=val[:, 0]
        y_no_L=val[:, 2]  
        df['Var1']=np.ravel(y_no_L)    
    
        
    # Make a quadratic function
    np.random.seed(seed)
    #X = time_rand1*np.random.normal(0, 1, nsim).reshape(nsim,1)         # Generate random numbers from a Uniform distribution 
    X= 8 * np.random.rand(nsim, 1)-3
    y_q =-0.9* X**2 +1.5*X+ 5+time_rand1*np.random.randn(nsim, 1)   # Add random noise from a Uniform distribution           
    df['Var2']=np.ravel(y_q)
    
    # Make a positive correlation
    np.random.seed(seed+30)    
    rand=a1*np.random.randn(nsim, 1)
    y_p=[-1 + b1 * X[i] + rand[i][0] for i in range(len(rand))]
    df['Var3']=np.ravel(y_p)
    
    # Make a negative correlation
    np.random.seed(seed+20)
    rand=a2*np.random.randn(nsim, 1)
    y_n=[20 - b2 * X[i] + rand[i][0] for i in range(len(rand))]    
    df['Var4']=np.ravel(y_n)
    
    # Make a negative correlation
    np.random.seed(seed+40)
    rand=a2*10*np.random.randn(nsim, 1)
    y_n=[1000 - b2 *0* X[i] + rand[i][0] for i in range(len(rand))]    
    df['Var5']=np.ravel(y_n)
    
    # Make a quadratic function
    np.random.seed(seed)
    #X = time_rand1*np.random.normal(0, 1, nsim).reshape(nsim,1)         # Generate random numbers from a Uniform distribution 
    X= 0.5 * np.random.rand(nsim, 1)-3
    y_q =0.6* X**2 +1.5*X+ 5+time_rand1*np.random.randn(nsim, 1)   # Add random noise from a Uniform distribution           
    df['Var6']=np.ravel(y_q)    
    
    df['Target']=np.ravel(X)
    return df         