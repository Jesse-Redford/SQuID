import streamlit as st

# General Dataprocessing
import numpy as np
import pandas as pd

# Classifcation Metrics and Classifiers
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import itertools

matplotlib.rcParams['font.family'] = 'serif' #"DejaVu Sans" #"Times New Roman"
plt.rcParams.update({'font.size': 16})
plt.rcParams['xtick.major.pad']='2'
plt.rcParams['ytick.major.pad']='2'
plt.rcParams['axes.labelpad']='2'

def replace_text(obj):
    if type(obj) == matplotlib.text.Annotation:
        txt = obj.get_text()
        txts = txt.strip().split("\n")
        if len(txts) == 4:
            obj.set_text(txts[0])
        else:
            obj.set_text(txts[-1])
    return obj


def split_data(df,split=(.5, .25, .25)):
    split = split
    training = split[0] + split[1]
    validation = (len(df) * split[1]) / ((training) * len(df))
    df["category"] = 'test'
    df_train = pd.concat(
        [df[df.label == class_label].sample(frac=training, replace=False, random_state=1) for class_label
         in df.label.unique()])
    df.loc[df_train.index, 'category'] = 'train'
    df_val = pd.concat([df[(df.label == class_label) & (df.category == 'train')].sample(
        frac=validation, replace=False, random_state=1) for class_label in df.label.unique()])
    df.loc[df_val.index, 'category'] = 'validate'
    return df

class SDT:
    def __init__(self, tabular_dataset=None,split=(.25,.25,.5)):
        self.tabular_dataset = tabular_dataset
        self.split = split

        self.dataset = split_data(self.tabular_dataset, split=self.split)
        self.selected_parameters = self.parameters = [ele for ele in self.dataset.columns if ele not in {'label', 'image', 'category'}]
        self.selected_labels = self.class_labels = self.dataset.label.unique()

        self.training_data = self.dataset.loc[self.dataset['category'].isin(['train', 'validate'])]
        self.testing_data = self.dataset.loc[self.dataset['category'].isin(['test'])]

        self.data = self.dataset.loc[self.dataset['category'].isin(['train', 'validate','test'])]


    def calculate_dprimes(self):
        dprime_scores = []
        combinations = []
        for feature in self.selected_parameters:
            dprime = 0
            class_dictionary = {}
            for i, label_i in enumerate(self.selected_labels[:-1]):
                for label_j in self.selected_labels[i + 1:]:
                    ui = self.training_data[self.training_data['label'] == label_i][feature].mean()
                    uj = self.training_data[self.training_data['label'] == label_j][feature].mean()
                    sigmai = self.training_data[self.training_data['label'] == label_i][feature].std()
                    sigmaj = self.training_data[self.training_data['label'] == label_j][feature].std()
                    dprime += np.abs((np.max([ui, uj]) - np.min([ui, uj])) / np.sqrt((sigmai ** 2 + sigmaj ** 2) / 2))
                    class_dictionary[label_i+'_vs_'+label_j] = np.abs((np.max([ui, uj]) - np.min([ui, uj])) / np.sqrt((sigmai ** 2 + sigmaj ** 2) / 2))
            combinations.append(class_dictionary)
            n = len(self.selected_labels)
            coeff = 1 / (np.math.factorial(n) / (np.math.factorial(2) * np.math.factorial(n - 2)))
            dprime_scores.append((coeff * dprime))


        self.dprime_df = pd.DataFrame(combinations,index = self.selected_parameters)

        # Extract best features from dprime matrix (take max along columns then demove duplicates)
        self.best_parameters = list(set([self.dprime_df[column].idxmax() for column in self.dprime_df]))

        best_parameters = []
        for column in self.dprime_df:
            best_parameters.append(self.dprime_df[column].idxmax())

        # choose what to sort dprime by, mean, std, min, max, %25 etc...
        df = self.dprime_df.apply(pd.DataFrame.describe, axis=1)['mean']
        ds = pd.DataFrame({'parameters':df.index, 'discriminability':df.values})
        return ds.sort_values(by='discriminability', ascending=False)

    def calculate_parameters_accuracy(self):
        results = []
        for selected_parameter in self.selected_parameters:
            threshold = float(self.training_data[selected_parameter].mean())
            # compute parmameter average of each class label and store as 2 list
            dd = self.training_data.groupby('label').agg({selected_parameter:['mean']}) # print('training sorted',dd[selected_parameter])
            train_labels = list(dd[selected_parameter].index)
            train_labels_means = list(dd[selected_parameter]['mean'])
            # create list of tuples (class_label,parmeter value) = ('class1',.5)...('classN',2.4)
            test_labels_and_values = list(zip(self.testing_data.label, self.testing_data[selected_parameter]))
            # Loop through test lave
            y_pred = []; y_true = []
            for test_label, test_value in test_labels_and_values:
                absolute_difference_function = lambda list_value: abs(list_value - test_value)
                closest_value = min(train_labels_means, key=absolute_difference_function)
                if test_value > threshold and len(self.selected_labels) == 2:
                    y_pred.append(train_labels[np.argmax(train_labels_means)])
                elif test_value < threshold and len(self.selected_labels) == 2:
                    y_pred.append(train_labels[np.argmin(train_labels_means)])
                else:
                    y_pred.append(train_labels[train_labels_means.index(closest_value)])
                y_true.append(test_label)
            acc = accuracy_score(y_true, y_pred)
            results.append((selected_parameter, acc))
        scores = pd.DataFrame(results, columns=['parameters', 'accuracy'])
        scores = scores.sort_values(by='accuracy', ascending=False)
        return scores.round(decimals=3)

    def update_dataset(self):
        self.data = self.data[self.data.label.isin(self.selected_labels) == True]
        self.training_data = self.training_data[self.training_data.label.isin(self.selected_labels) == True]
        self.testing_data = self.testing_data[self.testing_data.label.isin(self.selected_labels) == True]
        return None

    def dprime_vs_accuracy(self):
        self.update_dataset()
        return self.calculate_dprimes().merge(self.calculate_parameters_accuracy(), how='inner', left_on='parameters', right_on='parameters')

def plot_matrix(cm, classes, title):
    fig,ax = plt.subplots()
    ax = sns.heatmap(cm, cmap="jet", annot=True, xticklabels=classes, yticklabels=classes, cbar=False, fmt='.5g')
    ax.set(title=title, xlabel="Predicted Label", ylabel="True label")
    return fig

def compute_metrics(confusion_matrix=None,class_labels=None):
    #https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    cm = pd.DataFrame(confusion_matrix)
    # False positives
    FP = cm.sum(axis=0) - np.diag(cm)
    # False negatives
    FN = cm.sum(axis=1) - np.diag(cm)
    # True Positives
    TP = np.diag(cm)
    # True Negatives
    TN = cm.values.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    metrics ={  'ACC':ACC,
                'FP':FP,
                'FN':FN,
                'TP':TP,
                'TN':TN,
                'TPR':TPR,
                'TNR':TNR,
                'PPV':PPV,
                'NPV':FPR,
                'FNR':FNR,
                'FDR':FDR,}
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics['label'] = class_labels
    return df_metrics

# Maker Header for table information
st.title('SQuID')
st.subheader('Surface Quality and Inspection Descriptions')
st.write('Created by Jesse Redford')

uploaded_file = st.file_uploader("Upload Your Dataset In CSV Format (See Example for Proper Formatting)")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
else:
    n = 30
    np.random.seed(7)
    example_dataset = {'image': [f'image{i}.xyz' for i in range(n*4)],
                 'label': ['Non-Defective']*n + ['OrangePeel']*n + ['Scratches']*n + ['Pits']*n,
                 'Sa': list(np.random.normal(1, 1, n))+list(np.random.normal(5, 2, n))+list(np.random.normal(10, 2, n))+list(np.random.normal(15, 3, n)),
                 'Ssk': list(np.random.normal(0, .1, n))+list(np.random.normal(0, 2, n))+list(np.random.normal(1, 2, n))+list(np.random.normal(-1, 3, n)),
                 'OtherParameter': list(np.random.normal(10, 1, n))+list(np.random.normal(15, 2, n))+list(np.random.normal(5, 2, n))+list(np.random.normal(0, 3, n)),
                 }
    df = pd.DataFrame.from_dict(example_dataset)
    st.dataframe(df)

test_percentage = st.number_input('Define The Percentage of The Dataset You Want to Reserve For Testing', min_value=.05, max_value=1.0, value=.5, step=.05)
val_percentage = train_perentage = (1 - test_percentage) / 2

sdt = SDT(tabular_dataset=df,split=(train_perentage,val_percentage,test_percentage))
sdt.selected_labels = st.multiselect('Select Defect/Surface Categories You Want to Compare', list(sdt.class_labels), list(sdt.selected_labels))

for i,col in enumerate(st.columns(len(sdt.selected_labels))):
    label = sdt.selected_labels[i]
    train_examples = len(sdt.training_data[sdt.training_data["label"] == label])
    test_examples = len(sdt.testing_data[sdt.testing_data["label"] == label])
    col.write(f"{label}")
    col.write(f"{train_examples}/{test_examples} ")
    col.write(f"train/test")

if st.checkbox('Apply Feature Selection Algorithm To Automatically Select The Best Set of Parameters'):
    sdt.calculate_dprimes()
    sdt.selected_parameters = st.multiselect('Select The Parameters You Want to Analyze:', sdt.parameters, sdt.best_parameters)
    sdt.update_dataset()

else:
    sdt.selected_parameters = st.multiselect('Select the parameters to use for analysis and classification:',sdt.parameters,sdt.selected_parameters)
    # Make sure sdt class internally has updated dataset based on user selected labels and parameters
    sdt.update_dataset()

if st.checkbox('Single Descriptor Analysis'):
    selected_parameter = st.selectbox('Select a parameter to analyze:',sdt.selected_parameters)
    test_data, test_labels = map(list, zip(*[[sdt.testing_data.loc[sdt.testing_data['label'] == l][selected_parameter],l] for l in sdt.selected_labels]))
    data, labels = map(list, zip(*[[sdt.training_data.loc[sdt.training_data['label'] == l][selected_parameter],l] for l in sdt.selected_labels]))

    # Compute unbias threshold based on training data
    threshold = st.slider('threshold', min_value=float(sdt.training_data[selected_parameter].min()),
                                       max_value=float(sdt.training_data[selected_parameter].max()),
                                       value=float(sdt.training_data[selected_parameter].mean()))

    # Plot Distributions
    fig, ax = plt.subplots(figsize=(15, 10))
    palette = itertools.cycle(sns.color_palette())
    show_test_hist = st.checkbox('-overlay histogram data (test set)')
    show_train_hist = st.checkbox('-overlay histogram data (training set)')
    for i,d in enumerate(data):
        c = next(palette)
        sns.kdeplot(d, fill=True, label=labels[i], ax=ax,color=c)
        if show_train_hist:
            sns.histplot(data=d, label=labels[i], ax=ax, color=c)

    if show_test_hist:
        for i, d in enumerate(test_data):
            c = next(palette)
            sns.histplot(data=d, label=test_labels[i], ax=ax, color=c)

    ax.legend(title_fontsize='small',frameon=False)

    if st.checkbox('-overlay pdf of test data'):
        for i, d in enumerate(test_data):
            sns.kdeplot(d, fill=False, label=test_labels[i]+'(test)',linestyle='--', ax=ax)

    ax.axvline(threshold, 0, 1, color="k", linestyle="dashed", linewidth=1)
    min_ylim, max_ylim = plt.ylim()

    if len(sdt.selected_labels) <= 2:
        plt.text(threshold, max_ylim * 1.01, f"Threshold {threshold:.2f}", fontsize=25)

    ax.set(xlabel=selected_parameter+" Value (units)", ylabel="Density Estimation")
    ax.xaxis.get_label().set_fontsize(30)
    ax.yaxis.get_label().set_fontsize(30)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.legend(fontsize=30)
    st.pyplot(fig)

    # compute parmameter average of each class label and store as 2 list
    dd = sdt.training_data.groupby('label').agg({selected_parameter: ['mean']}) #print('training sorted',dd[selected_parameter])
    train_labels = list(dd[selected_parameter].index)
    train_labels_means = list(dd[selected_parameter]['mean'])

    # create list of tuples (class_label,parmeter value) = ('class1',.5)...('classN',2.4)
    test_labels_and_values = list(zip(sdt.testing_data.label, sdt.testing_data[selected_parameter]))

    # Loop through test lave
    y_pred = []
    y_true = []
    for test_label,test_value in test_labels_and_values:
        absolute_difference_function = lambda list_value: abs(list_value - test_value)
        closest_value = min(train_labels_means, key=absolute_difference_function)
        if test_value >= threshold and len(sdt.selected_labels)==2:
            y_pred.append(train_labels[np.argmax(train_labels_means)])
        elif test_value < threshold and len(sdt.selected_labels)==2:
            y_pred.append(train_labels[np.argmin(train_labels_means)])
        else:
            y_pred.append(train_labels[train_labels_means.index(closest_value)])
        y_true.append(test_label)

    cm = matrix = confusion_matrix(y_true, y_pred,labels=np.unique(y_pred)) #sdt.selected_labels) #df.label.unique())
    df_metrics = compute_metrics(confusion_matrix=cm,class_labels=np.unique(y_pred)) #sdt.selected_labels)
    correct = np.sum(np.diag(cm))

    if len(sdt.selected_labels) == 2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Parameter:", selected_parameter)
        col2.metric("Classification Method (Threshold):", "{:.2f}".format(threshold))
        col3.metric("# Correct Classifications: # Examples", str(correct) + ':' + str(np.sum(cm)))
        st.dataframe(df_metrics.set_index('label'))
    else:
        col1, col2, col3 = st.columns(3)
        col1.write("**Selected Parameter**: \n" +  selected_parameter)
        col2.write("**Classification Method**:  \n Nearest Mean")
        col3.write(f" **Correct: Total**  \n {correct}:{np.sum(cm)}")
        st.dataframe(df_metrics.set_index('label'))

if st.checkbox('Evaluate D-prime Matrix'):
    sdt.calculate_dprimes()
    co = st.slider(' d-prime cut off',0,7,0)
    df = sdt.dprime_df
    df[df < co] = np.nan

    fig = plt.figure(figsize=(len(df.columns), len(df)))
    sns.heatmap(df ,cmap='coolwarm', linewidths=0.5, annot=True,vmin=co)
    st.pyplot(fig)

if st.checkbox('Set Decision Tree Depth'):
    tree_depth = st.number_input('Set Max Depth of Decision Tree',value=len(sdt.selected_parameters),min_value=1, step=1)
else:
    tree_depth = None

if st.button('Train and Evaluate Decision Tree Classifer Based On Current Train/Test Split'):

    y_train = sdt.training_data['label'].to_numpy()
    x_train = sdt.training_data[sdt.selected_parameters].to_numpy()
    y_test = sdt.testing_data['label'].to_numpy()
    x_test = sdt.testing_data[sdt.selected_parameters].to_numpy()

    dt_clf = DecisionTreeClassifier(criterion = "entropy", random_state=77, max_depth=tree_depth).fit(x_train,y_train)

    fig, ax = plt.subplots(figsize=(20, 10))
    _ = tree.plot_tree(dt_clf, ax=ax,
                       feature_names=sdt.selected_parameters,
                       class_names=sdt.selected_labels,
                       filled=False, impurity=False, rounded=False, label='none', precision=2, fontsize=18)
    ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]
    st.pyplot(fig)

    cm = confusion_matrix(y_test, dt_clf.predict(x_test))
    confusion_fig = plot_matrix(cm, np.unique(y_test), f'Decision Tree Accuracy {np.sum(np.diag(cm)) / np.sum(cm)*100 :.02f} %')
    st.pyplot(confusion_fig)
    st.dataframe(compute_metrics(confusion_matrix=cm, class_labels=np.unique(y_test)).set_index('label'))

if st.button('Evaluate Classifer Using 5-Fold Validation Over A Varitey of Train/Test Splits'):
    # Define Classifers to Use
    models = {'classifers': ['Decision Tree'],
                      'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=77, max_depth=tree_depth),
                      'inputs': sdt.selected_parameters,
                      'x_data': sdt.data[sdt.selected_parameters].to_numpy(),
                      'y_data': sdt.data['label'].to_numpy()
                      }

    train_sizes = np.arange(.1,1,.1)
    colors = [ 'k', 'r', 'g']
    markers = ['o', 'x', 'd']
    n = 5
    fig = plt.figure(figsize=(10, 8))
    for i, classifer in enumerate(models['classifers']):
        avgs = [];stds = [];splits = []
        for test_percentage in train_sizes:
            clf = models[classifer]
            cv = StratifiedShuffleSplit(n_splits=n, test_size=test_percentage, random_state=0)
            scores = cross_val_score(clf, models['x_data'], models['y_data'], cv=cv)
            split = str(round(1-test_percentage,2)) + '/' + str(round(test_percentage,2))
            avgs.append(scores.mean()); stds.append(scores.std()); splits.append(split)
        plt.xticks(range(len(splits)), splits,rotation=45)
        plt.plot(range(len(splits)),avgs,label = classifer,marker=markers[i],markersize=15,linestyle='--',color='k')
        plt.errorbar(range(len(splits)), avgs, stds, capsize=10,color=colors[i],linestyle='None')
    plt.ylabel('Accuracy')
    plt.xlabel('Train / Test %')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.ylim(top=1)
    plt.locator_params(axis='y', nbins=20)
    st.pyplot(fig)