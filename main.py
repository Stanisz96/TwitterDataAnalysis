import library.fileoperations as fo



def main(step_number: int):
    # Testing
    if step_number == 0:
        print("Welcome to Twitter data analysis project!")

    # Restructure raw data to individual form and save them to feather format.
    # Where individual refers to data for one following user and 
    # all data of users that this user is following.
    if step_number == 1:
        fo.save_all_tweets_individuals()







if __name__=='__main__':
    main(1)