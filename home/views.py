from django.shortcuts import render

# Create the home view here : 
def home(request):
    return render(request, 'home/index.html')