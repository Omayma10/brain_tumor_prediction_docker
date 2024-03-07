import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def predict_brain_tumor(data):

    df = pd.read_csv("Brain_tumor_original.csv")
    df.drop("Image",inplace=True,axis=1)
    df.drop(['Mean',"Correlation","Coarseness"],axis=1,inplace=True)

    x = df.drop("Class",axis=1)
    y = df["Class"]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)


    df_new = data
    x_test_new = scaler.transform(df_new)

    tumor_model = joblib.load('random_forest_v2.joblib')
    y_prediction = tumor_model.predict(x_test_new)
    return y_prediction[0]

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/receive_values', methods=['POST'])
def receive_values():
    try:
        data = request.get_json()
        values = data.get('values')
        if values and len(values) == 10:
            print("Received values:", values)
            column_names = ['Variance', 'Standard Deviation', 'Entropy', 'Skewness', 'Kurtosis','Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity']
            data = pd.DataFrame([values], columns=column_names)
            prediction = predict_brain_tumor(data)
            if prediction == [0]:
                reponse = "no tumor"
            else:
                reponse = "tumor"
            return(reponse)
        else:
            return jsonify({'success': False, 'message': 'Invalid values'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
