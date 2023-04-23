from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
gauth = GoogleAuth()           
gauth.LocalWebserverAuth()       
drive = GoogleDrive(gauth)  

# Create a folder variable that will be uploaded to Google Drive under comp423-hw1 named expert_data

folder= []
folder_read = "/Users/berademirhan/Desktop/Comp423/expert_data/train/rgb/"
#read the files in the local folder
for filename in os.listdir(folder_read):
    f = os.path.join(folder_read, filename)

    print(f)
    #upload each i to MyDrive/comp423-hw1/expert_data/train/rgb
    gfile = drive.CreateFile({'parents': [{'id': '1bzD75Sx-iNstyF9Heunq2bqwVM-gXXD-'}]})
    gfile.SetContentFile(f)
    gfile.Upload()