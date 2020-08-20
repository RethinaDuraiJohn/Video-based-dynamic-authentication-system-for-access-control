from django.urls import path,include
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index,name='index'),
    url(r'^login/', views.user_login,name='user_login'),
    url(r'^dashboard/', views.dashboard,name='dashboard'),
    url(r'^request/', views.requestt,name='requestt'),
    url(r'^request_status/', views.request_status,name='request_status'),
    url(r'^awards/', views.awards,name='awards'),
    url(r'^logout/', views.user_logout, name='logout'),
    url(r'^signup/', views.authentication_view, name='signup'),
    url(r'^passvalidate/', views.passvalidate, name='passvalidate'),
    url(r'^expected/', views.expected, name='expected'),
    url(r'^friend/', views.friend, name='friend'),
    url(r'^myfriend/', views.myfriend, name='myfriend'),
    url(r'^myexpected/', views.myexpected, name='myexpected'),
    url(r'^myupdate/', views.myupdate, name='myupdate'),
    url(r'^verify_otp/', views.verify_otp, name='verify_otp'),
    url(r'^emg/', views.emg, name='emg'),

]
