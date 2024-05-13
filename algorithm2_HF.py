'''
Name   : Ashwin Sai C
Course : NLP - CS6320-001
Title  : Project 2 : Chatbot using Machine Learning
Term   : Spring 2024

'''

import yaml
import os
import numpy as np
from   collections import Counter
from   sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.pipeline import Pipeline
from   sklearn.neural_network import MLPClassifier
from   nltk.sentiment import SentimentIntensityAnalyzer
from   sklearn.metrics.pairwise import cosine_similarity
import random
from   autocorrect import Speller
import nltk
import requests
from   bs4 import BeautifulSoup
from   googlesearch import search
from   sklearn.neighbors import KNeighborsClassifier

bot_name          = "BlackBot"
sia_handle        = SentimentIntensityAnalyzer()
positive_comments = ["Thats wonderful!", "Reallyy great!", "Wow, thats cool!", "Awe struck! I am happyyy."]
negative_comments = ["Ah dont worry!", "You will like it as time progresses", "I got it, Astronomy is not for everyone!", "I will teach you, dont worry!"]
neutral_comments  = ["OK", "Its a tie, balance between both", "Not positive nor negative!"]

def web_search(sentence):

    # Define the search query
    query = sentence

    # Perform the web search and retrieve the URLs
    search_results = search(query, num=5, stop=5, pause=2)

    # Print the URLs
    for url in search_results:
        print(url)

        # Get the content of the first search result
        # first_url = next(url)
        response  = requests.get(url)
        soup      = BeautifulSoup(response.text, 'html.parser')

        # Find the main content of the page
        main_content = soup.find_all('p')

        # Extract and print the text
        text = '\n'.join([line.get_text() for line in main_content])
        print(text)

        break

def Response_Parsing(user_response,user):
    '''
        parameters  : user_response, knowledge_base_data, user 
        Description : This function is used to compare the user response with standard questions
                      and if no match is found, it parses the user response and performs NLP 
                      techniques and returns the best fit response

        return      : model response
    '''
    user_data        = Load_User_Model(user)
    response_list    = []
    user_response    = user_response.lower().replace("blackhole","black hole")
    response_output  = "" 
    random_greeting  = ["Howdyy!","Good day!","Yo man!", "Good day ladies and gentlemen!", "Welcome to Space!"]

    if "hey" in user_response.split(" ") or "greetings" in user_response.split(" ") or "hi" in user_response.split(" ") or "whats up" in user_response.split(" "):
        response_output += str(random_greeting[random.randint(0,4)])
        response_output += "\nHey!! I am "+bot_name

    elif ("who am i" in user_response.lower()) or ("what is my name" in user_response.lower()) :
        response_output += "How can I forget "+user+", ;)"
        
    elif "what is your name" in user_response or "can you tell me your name" in user_response or Cosine_Simmilarity_Function(user_response, "who are you") > 0.30:
        response_output += "My name is "+bot_name

    elif "my name is" in user_response:
        response_output += "Ofcourse! I know you. (See the chat name)"

    elif "do you believe in god" in user_response:
        response_output += "Ashwin Sai created me, so yes!"

    elif "about me" in user_response or Cosine_Simmilarity_Function("what do you know about me", user_response) >= 0.25:
        response_output += "\nYour name is "+user_data['Name']
        response_output += "You like "+user_data['FavColor'].replace("\n","")+" color right?"
        response_output += "You stay in "+user_data['Location']
        response_output += "The weather there is "+user_data['weather']
        response_output += "You are currently "+str("Single" if user_data["Marital"].lower() == "yes" else "Married")
        response_output += " And you are a " +str("Student" if user_data["Job"].lower() == "yes" else "Working")

    elif "fav color" in user_response or Cosine_Simmilarity_Function("favourite color", user_response) >= 0.25:
        response_output += "\nYou like "+user_data['FavColor'].replace("\n","")+" color right?"

    elif "where do I stay" in user_response or Cosine_Simmilarity_Function("where am I", user_response) >= 0.20:
        response_output += "\nYou stay in "+user_data['Location']

    elif "how is the weather" in user_response:
        response_output += "\nThe weather there is "+user_data['weather']

    elif "what am I" in user_response:
        response_output += "\nYou are currently "+user_data['Marital']
        response_output += "\nAnd you are a " +str("Student" if user_data["Job"].lower() == "yes" else "Working")   

    elif "who built you" in user_response or "owner" in user_response or "author" in user_response:
        response_output += "Ashwin Sai is the one!"

    elif "thank you" in user_response:
        response_output += "Sure Glad, I could help you! Anytime!"

    elif "what do you know" == user_response:
        response_output += "Hmm, to be honest I know only about Black Holes right now. I can help you with that."

    elif "bye" in user_response or "see you soon" in user_response:
        print("\n"+bot_name+": Did you learn something today?\n")
        end_senti = input(user+":")
        sentiment_score = sia_handle.polarity_scores(end_senti)

        if sentiment_score['compound'] >= 0.05:
            response_output += positive_comments[random.randint(0,3)]+"\nSee you soon! "+str(user)
        elif sentiment_score['compound'] <= 0.05:
            response_output += negative_comments[random.randint(0,3)]+"\nSee you soon! "+str(user)
        else:
            response_output += neutral_comments[random.randint(0,2)]+"\nSee you soon! "+str(user)

    elif "how can you help me" in user_response:
        response_output += "I cant fix your car nor cook food, but I can help you about the topic 'Black hole' ;)"

    else:
        response_output = "None"
        
        return response_output

    return response_output

def load_conversation_history(user):

    file_handle = open("user_model_"+user+".txt","r", encoding='utf-8')
    data        = file_handle.readlines()
    file_handle.close()

    for line in data:
        print(line)   

def read_yaml(file_path, related_file):
    questions  = []
    answers    = []

    with open(file_path,"r") as handle:
        data = handle.readlines()

    for line in data:
        if "- question:" in line:
            questions.append(line.replace("- question:","").strip())
        if "  answer:" in line:
            answers.append(line.replace("  answer:","").strip())

    # print("Dataset loaded..")

    with open(related_file,"r") as handle:
        data = handle.readlines()

    for line in data:
        if "- question:" in line:
            questions.append(line.replace("- question:","").strip())
        if "  answer:" in line:
            answers.append(line.replace("  answer:","").strip())

    # print("Dataset loaded..")

    return [questions, answers]

def train_KNN(questions, answers):
   # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
        ('knn', KNeighborsClassifier(n_neighbors=7))  # KNN classifier with k=1
    ])

    print("loading KNN model..")
    # Train the model
    pipeline.fit(questions, answers)

    return pipeline

def train_model(questions, answers):
    # print("Starting pipeline..")

    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
        ('clf', MLPClassifier(hidden_layer_sizes=(100,50,), max_iter=1000))
    ])

    print("loading MLPC Model..")

    # Train the model
    pipeline.fit(questions, answers)

    KNN_model = train_KNN(questions, answers)

    return pipeline, KNN_model

def get_response(pipeline, new_question, KNN_model):

    predicted_answer  = pipeline.predict([new_question])

    predicted_answer2 = KNN_model.predict([new_question])

    # print("KNN  : ", predicted_answer2)
    # print("MLPC : ", predicted_answer)

    KNN_sim  = Cosine_Simmilarity_Function(predicted_answer2[0], new_question)
    MLPC_sim = Cosine_Simmilarity_Function(predicted_answer[0], new_question)

    # print(KNN_sim, MLPC_sim)

    if KNN_sim > MLPC_sim:
        return predicted_answer2
    else:
        return predicted_answer
        
def chatbot_bot(pipeline):

    while True:
        # Example inference
        new_question = input("Question: ")
        if new_question == "exit":
            exit(0)

        response = get_response(pipeline, new_question)
        print("Chabot: ",response)

def Initial_User_Data(name):
    '''
        parameters  : user name 
        Description : This function is used to gather initial details about the user for creating user model

        return      : user name 
    '''

    colors            = ["red","blue","white","grey","green"]
    stayy             = {"chatham":"Holy coww! A brave person!!","northside":"Lucky!!"}
    weather           = ["hot","cold"]
    current_weather   = "warmyy"
    flag              = 0   

    print("---------------------------------------------------")
    print("          Black Coffee with the Black Bot v2.0     ")
    print("---------------------------------------------------")
    print("Greetings! I am back.\nHowwdyy my name is "+bot_name+", Let me explain about myself,")
    print("I am studying my M.S. about Black Hole at Hogwarts Spring 2024.")
    print("I am good with Information but at the same time terrible at Informationnn tooo !!!!")
    print("Please dont get angry about how this convoo goesss.")
    
    print("\n"+bot_name+": Now Lets start with you, What is your good name?\n")
    user_name = input(name+": ")
    if user_name.lower() == "q":
        print(bot_name+": Ahh too soon! bye-bye")
        exit(0)
    user = user_name

    if os.path.exists("user_model_"+user_name.replace(" ","_").lower()+".txt"):
        print("Welcome back ",user_name)
        flag = 1
    else:
        print("\n"+bot_name+": What is your favourite color?\n")
        user_color = input(user+":")
        if user_color.lower() == "q":
            print(bot_name+":Ahh too soon! bye-bye")
            exit(0)
        if user_color.lower() in colors:
            print(bot_name+": ",user_color," is an awesome color!")
        elif user_color.lower() == "black":
            print(bot_name+": Thatss my fav color too yayy!!")
        else:
            print(bot_name+": Not bad, good choice!")

        print("\n"+bot_name+": Where do you stay currently?\n")
        user_loc = input(user+":")
        if user_loc.lower() == "q":
            print(bot_name+": Ahh too soon! bye-bye")
            exit(0)
        try:
            print(bot_name+": ",stayy[user_loc.lower()])
        except Exception as e:
            print(bot_name+": ",user_loc," is a popular place!")

        print("\n"+bot_name+": How is the weather outside?\n")
        user_weather = input(user+":")
        if user_weather.lower() == "q":
            print(bot_name+":Ahh I just askedd the weatherrr! bye-bye")
            exit(0)
        if user_weather in weather:
            print(bot_name+": Damm, its extreme right?")
        else:
            print(bot_name+": hmm, must be a nice weather!")

        print("\n"+bot_name+": Are you single? :)\n")
        user_marital = input(user+":")
        if user_marital.lower() == "q":
            print(bot_name+": Nooooo! Dont gooo yet! :(")
            exit(0)
        if user_marital.lower() == "yes":
            print(bot_name+": Awesome! Me too!!")
        else:
            print(bot_name+": Woww, thats cool, have fun together!")

        print("\n"+bot_name+": Are you a student?\n")
        user_job = input(user+":")
        if user_job.lower() == "q":
            print(bot_name+": bye-bye")
            exit(0)
        if user_job.lower() == "yes":
            print(bot_name+": Studyy well, All the best!")
        else:
            print(bot_name+": Go become a student, its fun!")

        print("\n"+bot_name+": How much do you like about space (galaxy, stars, astronomical objects)?\n")
        user_likes = input(user+":")
        sentiment_score = sia_handle.polarity_scores(user_likes)

        if sentiment_score['compound'] >= 0.05:
            print(positive_comments[random.randint(0,3)])
        elif sentiment_score['compound'] <= 0.05:
            print(negative_comments[random.randint(0,3)])
        else:
            print(neutral_comments[random.randint(0,2)])


    if flag == 0:
        User_Model_Creation(user_name,user_color,user_loc,user_weather,user_marital,user_job,user_likes)

    if "yes" == input("Do you want me to load our conversations? (yes/no) :") and os.path.exists('user_model_'+user+'.txt'):
        load_conversation_history(user_name)

    print("\n\n"+bot_name+": Now lets get down to business, How can I help you?")


    return user_name

def User_Model_Creation(user_name,user_color,user_loc,user_weather,user_marital,user_job,user_likes):
    '''
        parameters  : user_name,user_color,user_loc,user_weather,user_marital,user_job 
        Description : This function is used to create a useer model

        return      : None 
    ''' 

    file_handle = open("user_model_"+user_name.replace(" ","_").lower()+".txt","w")

    file_handle.write("Name : "+str(user_name)+"\n")
    file_handle.write("FavColor : "+str(user_color)+"\n")
    file_handle.write("Location : "+str(user_loc)+"\n")
    file_handle.write("weather : "+str(user_weather)+"\n")
    file_handle.write("Marital : "+str(user_marital)+"\n")
    file_handle.write("Job : "+str(user_job)+"\n")
    file_handle.write("Likiness_about_Space : "+str(user_likes)+"\n")

    file_handle.close()

    user = user_name
    print("User Model created!")

def Load_User_Model(user):
    '''
        parameters  : user 
        Description : This function is load the user model

        return      : user model data 
    ''' 
    data_dict    = {}
    try:
        file_handle = open("user_model_"+user.replace(" ","_").lower()+".txt","r", encoding='utf-8')
        data        = file_handle.readlines()
        file_handle.close()

        for line in data:
            try:
                data_dict[line.split(":")[0].replace(" ","")] = line.split(":")[1]
            except Exception as e:
                pass

        return data_dict

    except Exception as e:
        #print(e)
        print("User "+user+" profile not present!")

def User_Model_Updation(data,user):
    '''
        parameters  : data , user 
        Description : This function is used to update the user model with the uesr's chat history

        return      : None 
    ''' 

    file_handle = open("user_model_"+user+".txt","a", encoding='utf-8')
    file_handle.write("\n")
    for item in data:
        file_handle.write(item)
        file_handle.write("\n")

    file_handle.close()

    print("User model updated!")
    print("Name : ",user)

def Add_to_questions(user_input, model_response):
    
    file_handle = open("related_questions.yaml","a")
    file_handle.write("- question: "+str(user_input))
    file_handle.write("\n")
    file_handle.write("  answer: "+str(model_response))
    file_handle.write("\n\n")
    file_handle.close()    

def Chatbot(pipeline, KNN_model):
    '''
        parameters  : None
        Description : This function is initiates the Chatbot

        return      : None 
    ''' 

    # knowledge_base_data = Load_Pickle("DK_Base.pickle")
    user                = "unknown"
    user                = Initial_User_Data(user)   
    user_interactions   = []
    few_topics          = ["Infinite Density", "Event Horizon", "Formation","No Light", "Time Dilation", "Hawking Radiation",\
                           "Spaghettification", "Size", "Jet Formation", "Cosmis influence", "Black Hole"]

    while True:
        # try:
        user_input     = input(user+": ")
        if user_input.lower() == "q":
            print("Thank you for chatting, See you soon!")
            break

        user_input     = autocorrect_sentence(user_input)
        model_response = Response_Parsing(user_input, user)

        if model_response == "None":
            model_response = get_response(pipeline, user_input, KNN_model)            
            model_response = str(" ".join(model_response))

        print("Found results matching for '"+user_input+"'")
        print(bot_name,": ",model_response)
        user_interactions.append(user+": "+user_input)
        user_interactions.append(bot_name+": "+model_response)
        print("\n")
        
        if "yes" == input("\nWas is relatable? (yes/no): ").lower():
            Add_to_questions(user_input,model_response)
        else:
            if "yes" == input("\nDo you want me to check the internet about this? (yes/no) : "):
                web_response = web_search(user_input)
                print(web_response)
                if "yes" == input("\nWas is relatable? (yes/no): ").lower():
                    Add_to_questions(user_input,model_response)


        # except Exception as e:
            # print(e)
            #print("I can help you out with these topics too, " + "\n".join(few_topics) +" \n")

    User_Model_Updation(user_interactions,user)

def sentence_to_bow(sentence):

    return Counter(sentence.split())

def Cosine_Simmilarity_Function(user_response,word):

    # Convert sentences to Bag of Words representations
    bow1 = sentence_to_bow(user_response)
    bow2 = sentence_to_bow(word)
    
    # Get the set of all unique words in both sentences
    words = set(bow1.keys()).union(set(bow2.keys()))
    
    # Create vectors for each sentence
    vector1 = np.array([bow1.get(word, 0) for word in words])
    vector2 = np.array([bow2.get(word, 0) for word in words])
    
    # Calculate dot product
    dot_product = np.dot(vector1, vector2)
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(vector1)
    mag2 = np.linalg.norm(vector2)
    
    # Calculate cosine similarity
    similarity = dot_product / (mag1 * mag2)

    # print(similarity)
    
    return similarity

def autocorrect_sentence(sentence):

    spell        = Speller(lang="en")
    correct_sent = " ".join(spell(word) for word in sentence.split())

    return correct_sent

def Initialize_Chatbot():
    file_path            = "responses.yaml"
    related_file         = "related_questions.yaml"
    [questions, answers] = read_yaml(file_path,related_file)
    pipeline, HMM_model  = train_model(questions, answers)

    Chatbot(pipeline, HMM_model)

if __name__ == "__main__":

    Initialize_Chatbot()