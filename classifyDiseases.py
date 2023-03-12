import re
import json
import pandas as pd
import openai
import string
import requests
import random
import numpy as np
import pandas as pd
import pickle
import os
import openai as ai
import pickle

labels = ["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", 
          "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", 
          "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety", 
          "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", 
          "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", 
          "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", 
          "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", 
          "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", 
          "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach", 
          "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation", 
          "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", 
          "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", 
          "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs", 
          "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", 
          "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", 
          "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", 
          "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", 
          "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_ofurine", 
          "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)", 
          "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body",
          "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", 
          "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", 
          "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", 
          "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen", 
          "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf", 
          "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling", 
          "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister", "red_sore_around_nose", 
          "yellow_crust_ooze"]
labels_str = ', '.join(labels)


def getsymptomList(userText):    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {'role': 'system', 
        'content': f'You are a helpful medical assistant. When the user describes their symptoms, you identify which of the labels listed below best describes thier symptoms. Here is the list of allowed labels: {labels_str}'},
        {'role': 'user',
        'content': "I've been itching all over like crazy for the past few days, and it's driving me insane."},
        {'role': 'assistant',
        'content': 'itching'},
        {'role': 'user',
        'content': "My knee is really stiff and it hurts when I try to straighten it."},
        {'role': 'assistant',
        'content': 'knee_pain'},
        {'role': 'user',
        'content': "My knee is swollen and tender to the touch, and it hurts when I put weight on it."},
        {'role': 'assistant',
        'content': 'knee_pain'},
        {'role': 'user',
        'content': "My cramps feel like a constant ache that doesn't go away."},
        {'role': 'assistant',
        'content': 'cramps'},
        {'role': 'user',
        'content': "I've been seeing flashes of light or spots in my vision that aren't really there."},
        {'role': 'assistant',
        'content': 'visual_disturbances'},
        {'role': 'user',
        'content': "My vision seems dim or hazy, like looking through a foggy window."},
        {'role': 'assistant',
        'content': 'visual_disturbances'},
        {'role': 'user',
        'content': userText}
        ]
    )
    inputList = response['choices'][0]['message']['content'].split(', ')
    return inputList

def develop_inputList(diseaseList):
    values = [0 for i in labels]
    for i in diseaseList:
        index = labels.index(i)
        values[index] = 1
    return values

def getInfo(disease):
    model_engine = "text-davinci-003"
    prompt = f"Here is a disease {disease}, provide valuable information about it"
    completion = openai.Completion.create(
                                        engine=model_engine,
                                         prompt=prompt,
                                         max_tokens=1024,)
    return completion.choices[0].text
