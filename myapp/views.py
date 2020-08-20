from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render,get_object_or_404, redirect
from django import template
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate,login,logout
from datetime import datetime
from myapp.forms import Authentic
from django.conf import settings
import cv2
import os
from .forms import *
from .models import *
from django_otp.oath import totp
import time
import base64
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from scipy import misc
import cv2
import numpy as np
from PIL import Image
import glob
from collections import defaultdict
from myapp import facenet
from myapp import detect_face
import os
import time
import pickle
from PIL import Image
import glob
from twilio.rest import Client
account_sid = 'AC730d2142614ea4d8220278bb5bd247fd'
auth_token = 'fdd8833312fe3d2ebc9eb7142c3cb771'
##################################################################
# account_sid = 'ACf896cacf87342b00d224a999be391e7e'
# auth_token = 'c26c67fbb8193ebb4cec3a8364469a4a'
client = Client(account_sid, auth_token)

def sms(msg, phn_number):
	message = client.messages \
	                .create(
	                     body=str(msg),
	                     from_='+15093977702',
	                     # from_='+15017122661',
	                     to='+91'+ str(phn_number)
	                 )

	print(message.sid)

def call(phn):
	call = client.calls.create(
	                        twiml='<Response><Say>Alert!, You have a request</Say></Response>',
	                        to='+91'+str(phn),
	                        from_='+15093977702'
	                    )

	print(call.sid)


secret_key = b'12345678901234567890'
now = int(time.time())
modeldir = 'myapp/model/20170511-185253.pb'
classifier_filename = 'myapp/class/classifier.pkl'
npy='myapp/npy'
train_img="/myapp/train_img"
with tf.Graph().as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160

                #HumanNames = os.listdir(train_img)
                #HumanNames.sort()
            HumanNames = ['ilaya','shanmu','shanu']
            print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)



#from .moimdels import Document
# Create your views here.

def index(request):
    if request.user.is_authenticated:
        if request.user.is_superuser:
            return render(request,'security.html',)
            #return HttpResponseRedirect('/admin/')
        else:
            #pass
            return render(request,'inmates.html',)
            #rreturn HttpResponseRedirect('index')
    else:
        return render(request,'index.html',)

@login_required
def dashboard(request):
    if request.user.is_superuser:
        return render(request,'security.html',)
    else:
        return render(request,'inmates.html',)


@login_required
def myupdate(request):
    if request.user.is_superuser:
        pass
    else:
        return HttpResponse("updation", content_type='text/plain')





@login_required
def verify_otp(request):
    if request.user.is_superuser:
        if request.method =='POST':
            username=request.POST.get("username")
            print (username)
            password=request.POST.get("password")
            try:
                originaluser = otp.objects.filter(name=username)
                length = len(originaluser)
                print(length)
                originaluser= originaluser[length-1]
                if originaluser:
                    if originaluser.otp == password:
                        return HttpResponse("verified and allow")
                    else:
                        return HttpResponse("invalid OTP")
            except:
                return HttpResponse("invalid name")


        return render(request,'verify_OTP.html',)
    else:
        return HttpResponse("OTP Verify", content_type='text/plain')


def modelload():
    pass


@login_required
def expected(request):
    if request.user.is_superuser:
        return HttpResponse("NOT VALID", content_type='text/plain')
    else:
        if request.method == 'POST':
            form = expectedvis(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                return render(request,'inmates.html')
        else:
            form = expectedvis()


        return render(request,'exp.html',{'form' : form})
        #return render(request,'exp.html',)

@login_required
def friend(request):
    if request.user.is_superuser:
        return HttpResponse("NOT VALID", content_type='text/plain')
    else:
        if request.method == 'POST':
            form = friendvis(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                return render(request,'inmates.html')
        else:
            form = friendvis()


        return render(request,'friend.html',{'form' : form})




@login_required
def myfriend(request):
        if request.user.is_superuser:
            return HttpResponse("NOT VALID", content_type='text/plain')
        else:
            if request.method == 'GET':
                username = request.user.username
                Image2 = friendvisitor.objects.filter(user = username )
                stu = {"details": Image2 }
                return render(request,'myfriend.html',stu)

                if request.method == 'POST':
                    if 'remove' in request.POST :
                        team = friendvisitor.objects.get(id=request.POST.get("student_id"))
                #print (team)
                    team.delete()

            return HttpResponse("Deleted", content_type='text/plain')


@login_required
def myexpected(request):
    if request.user.is_superuser:
        return HttpResponse("NOT VALID", content_type='text/plain')
    else:
        if request.method == 'GET':
            username = request.user.username
            Image2 = expectedvisitor.objects.filter(user = username )
            stu = {"details": Image2 }
            return render(request,'myexpected.html',stu)
        elif request.method == 'POST':
            if 'send OTP' in request.POST :
                team = expectedvisitor.objects.get(id=request.POST.get("student_id"))
                name = team.name
                number = team.phone
                visiname = request.user.username
                otplist = []
                for delta in range(10,110,20):
                    otplist.append(totp(key=secret_key, step=10, digits=6, t0=(now-delta)))
                otpnumber= otplist[0]
                sms (str(otpnumber),number)
                print (otpnumber)
                form = otp.objects.create(name=name,otp=otpnumber, visiname = visiname,phone = number)
                form.save()
                #print("hai")

                message = f"OTP is sent to {number}"
                return HttpResponse(message, content_type='text/plain')
            elif 'Resend OTP' in request.POST :
                team = expectedvisitor.objects.get(id=request.POST.get("student_id"))
                number = team.phone
                message = f"OTP is resent to {number}"
                return HttpResponse(message, content_type='text/plain')

        #return HttpResponse("OTP sent to ", content_type='text/plain')

def test():
    pass



@login_required
def awards(request):
    listdetails =[]
    # cam = cv2.VideoCapture(0)

    # cv2.namedWindow("test")

    # img_counter = 0

    # while True:
    #     ret, frame = cam.read()
    #     cv2.imshow("test", frame)
    #     if not ret:
    #         break
    #     k = cv2.waitKey(1)

    #     if k%256 == 27:
    #         # ESC pressed
    #         print("Escape hit, closing...")
    #         break
    #     elif k%256 == 32:
    #     # SPACE pressed
    #         path = r'C:\Users\Lenovo\Desktop\images\new visitor'
    #         #cv2.imwrite(os.path.join(path , 'photo.jpg'), frame)
    #         img_name = "photo.jpg"
    #         os.chdir(path)
    #         cv2.imwrite(img_name, frame)
    #         #print("{} written!".format(img_name))
    #         #img_counter += 1

    #         cam.release()

    #         cv2.destroyAllWindows()
    #         break


    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            phn_number = '7639147936'
            call(phn_number)
            # return render(request,'security.html',)
            return redirect("/request_status/")
    else:
        form = UploadForm()


    return render(request,'awards.html',{'form' : form})





@login_required
def requestt(request):
    flag = 1
    final= []
    video_capture = cv2.VideoCapture(0)
    c = 0
    check = 1
    id = 0
    print('Start Recognition')
    prevTime = 0
    while True:
        if(check):
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                    # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        # print("predictions")
                        print(best_class_indices,' with accuracy ',best_class_probabilities)

                    # print(best_class_probabilities)
                        if best_class_probabilities>0.9:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            print('Result Indices: ', best_class_indices[0])
                            print(HumanNames)
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    #s= ''
                                    #s = result_names+' with accuracy '+best_class_probabilities
                                    #final.append(s)
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                            flag = 1
                            check = 0
                            break
                        else:
                            path = "A:/github/sih/media/"

                            cv2.imwrite(os.path.join(path,"frame%d.jpg" % id),frame)
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                        #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            result_names = "unknown"

                            #final.append('unknown')
                            cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)
   
                            id+=1
                            flag = 2
                            check = 0
                            call(7092600127)
                            break
                            
                            

                else:
                    print('Alignment Failure')

            cv2.imshow('Video', frame)

        else:
            break




    video_capture.release()
    cv2.destroyAllWindows()
    if(flag == 1):
        return render(request,'request.html',)
    elif(flag == 2):
        return redirect("/awards/")


@login_required
def emg(request):
    if request.user.is_superuser:

        image_list = []
        for i in os.listdir("A:/github/sih/media/"):
            if(i!="images"):
                image_list.append(str(i))

        return render(request,'emg.html',{'img':image_list})



@login_required
def request_status(request):
    if request.user.is_superuser:
        if request.method == 'GET':
            Image = scanupload.objects.all()
            Image = Image.order_by('-id')
            #print (Image)
            #print(type(Image))

            stu = {"details": Image }
            return render(request,'request_status.html',stu)
    else:
        if request.method == 'POST':
            if 'reject' in request.POST :
                team = scanupload.objects.get(id=request.POST.get("student_id"))
                if team.status == 'allow':
                    team.status = 'pending'
                team.save(update_fields=['status'])
                team.save()

            elif 'accept' in request.POST :
                team = scanupload.objects.get(id=request.POST.get("student_id"))
                if team.status == 'pending':
                    team.status = 'allow'
                team.save(update_fields=['status'])
                team.save()

            elif 'Add as friend' in request.POST:
                team1 = scanupload.objects.get(id=request.POST.get("student_id"))
                name = team1.name
                number = team1.phone
                image = team1.upload_Main_Img
                form1 = friendvisitor.objects.create(name=name,phone = number, image = image)
                form1.save()
                return HttpResponse("added", content_type='text/plain')


        else:
            username = request.user.username
            Image1 = scanupload.objects.filter(visiname = username )
            Image1 = Image1.order_by('-id')
            stu1 = {"details": Image1}
            print (stu1)
            return render(request,'Request_handling.html',stu1)


        #return render(request,'Request_handling.html',stu1)
        return HttpResponse("okay", content_type='text/plain')




@login_required
def user_logout(request):
    # instance = scanupload.objects.all()
    # instance.delete()
    logout(request)
    return HttpResponseRedirect(reverse('index'))


@login_required
def passvalidate(request):
        if request.method=="POST":
            username=request.POST.get("username")
            password=request.POST.get("password")

            user=authenticate(username=username,password=password)

            if user:
                print ("password validation is succesful")
                return HttpResponse("password validation is succesful", content_type='text/plain')

            else:
                print("invalid username and password")
                return HttpResponse("wrong credantials", content_type='text/plain')

        else :
            return render(request,'passvalidate.html',)



# Create your views here.
def user_login(request):
    if request.user.is_authenticated:
            return HttpResponseRedirect(reverse('index'))
    else:
        if request.method=="POST":
            username=request.POST.get("username")
            password=request.POST.get("password")

            user=authenticate(username=username,password=password)

            if user:
                login(request,user)
                return HttpResponseRedirect(reverse('index',))
            else:
                return HttpResponse("invalid username and password")
        else :
            return render(request,'login.html',)


def authentication_view(request):
    registered = False

    if request.method=="POST":
        print(request.POST)
        auth=Authentic(request.POST )

        if auth.is_valid():
            auth=auth.save(commit=False)
            auth.set_password(auth.password)
            #hashing the password
            auth.save()
            registered=True
        else :
            print("error")
    else:
        auth=Authentic()
    return render(request,'login.html',)
