from tensorflow.keras.models import load_model


def do_primary_prediction(spectrogram):
    loaded_model = load_model('./models/binary_model_v2.h5')

    result = loaded_model.predict(spectrogram, batch_size=1)

    if result[0][0] > 0.5:
        return {'result': False, 'probability': result[0][0]}
    else:
        return {'result': True, 'probability': result[0][0]}


def do_secondary_prediction(spectrogram):
    loaded_model = load_model('./models/disease_model_v1.h5')
    result = loaded_model.predict(spectrogram)

    # Sort the array in descending order
    sorted_numbers = sorted(result[0], reverse=True)
    # Get the top 3 values
    top3_values = sorted_numbers[:3]
    # Get the indexes of the top 3 values
    indexes_of_top3_values = sorted(range(len(result[0])), key=lambda i: result[0][i], reverse=True)[:3]
    
    disease_list = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', "Bronchitis", 'COPD', 'Lung Fibrosis',
                    'Pleural Effusion', 'Pneumonia', 'URTI']
    predicted_disease = {}
    diseases = []
    for i in range(3):
        print(i)
        disease = disease_list[indexes_of_top3_values[i]]
        diseases.insert(i, disease)

    predicted_disease['diseases'] = diseases
    predicted_disease['probabilities'] = top3_values

    return predicted_disease