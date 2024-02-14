from cmath import nan
import pandas as pd

#PREPROCESSING
#PRIMA DI CREARE IL DIZIONARIO COMPLETO, CI ASSICURIAMO DI RIMUOVERE I SOTTOGENERI CHE NON RAPPRESENTANO UNA SFIDA NELL'AMBITO.
#IN PARTICOLARE, RIMUOVIAMO TUTTI I SOTTOGENERI LEGATI ALLE NAZIONALITA'. NONOSTANTE LE NAZIONALITA' ABBIANO INFLUENZE CULTURALI, QUESTI PATTERN RISULTEREBBERO TROPPO COMPLESSI 
#E RICHIEDEREBBERO UN AMMONTARE DI DATI E DI RISORSE COMPUTAZIONALI TROPPO ELEVATO
#REMOVE NATIONALITY PREFIX (THERE'S A FUNCTION FOR PROCESSING SOME GENRE)
def remove_nationality_prefix(genres, nationality_list):
        updated_genres=[]
        for genre in genres:
            for nationality in nationality_list:
                if nationality in genre:
                    #Remove prefix from nationality, considering the presence of -
                    genre = genre.replace(nationality, '').replace('-', '').strip()
            if 'j-rock' in genre or 'k-rock' in genre:
                genre = 'rock'
            elif 'j-pop' in genre:
                genre='pop'
            elif 'korean pop' in genre:
                genre= 'k-pop'
            elif 'j-metal' in genre:
                genre='metal'
            elif 'v-pop' in genre:
                genre='pop'
            elif 'j-punk' in genre:
                genre='punk'
            elif 'c-pop' in genre:
                genre='pop'
            elif 'edm' in genre or 'techno' in genre or 'house' in genre:
                genre='electronic'
            updated_genres.append(genre)
        return updated_genres
#REMOVE SUBGENRE THAT HAVE LESS THAN 10 ISTANCE OR THAT ARE TOO DEEP AND CANT REPRESENT THE GENRE
def remove_subgenre(genres, subgenre_remove):
    found = False

    for genre in genres:
        for subgenre in subgenre_remove:
            if subgenre in genre:
                found = True
                break  
        if found:
            break  

    return found

#DIVIDE SUBGENRE THAT INCLUDE MORE THAN ONE MAIN GENRE (POP ROCK -> POP AND ROCK)
def divide_and_add_subgenres(genres, subgenre_divide):
        updated_genres = []
        for genre in genres:
            if genre in subgenre_divide:           
                updated_genres.extend(genre.split())
            else:
                updated_genres.append(genre)
        return updated_genres
  
def preprocess(df,nationality,subgenre_remove,subgenre_divide,main_genres,popularity):
    #filter for choose the dimension of the dataset
    df=df[df['popularity']>popularity]
    df.drop(columns=['popularity'],inplace=True)
    indexes=[]
    for index,row in df.iterrows():
        genres=row['genres'].split(',')
        remove=remove_subgenre(genres,subgenre_remove)
        if remove == True:
           indexes.append(index)
    df=df.drop(indexes)
    for index,row in df.iterrows():
        genree=[]
        genres=row['genres'].split(',')
        genres = remove_nationality_prefix(genres,nationality)
        genres = divide_and_add_subgenres(genres,subgenre_divide) 
        genres=list(set(genres))
        #MAP SUBGENRES IN MAIN GENRES
        for subgenre in genres:
            trovato=False
            if 'hip hop' in subgenre:
                subgenre = 'rap'
            for main in main_genres:
                if main in subgenre:
                    subgenre = main
                    trovato=True 
            
            if trovato==True:
                genree.append(subgenre)

        genree=list(set(genree))

        if not genree:
            df.drop(index, inplace=True)
        else:
            df.at[index,'genres']=genree
    main_genres.remove('hip hop')
    return df


    