import json

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .ML.utils import generateArticle, makeGenerationResponse, makeGenerationResponseMobile

from PIL import Image

def index(request):
    return render(request, 'api/spa.html')

def test(request):
    return HttpResponse("return this string")

@csrf_exempt
def generate(request):
    if request.method == 'POST':
        config_data = request.POST.get('config', None)
        if config_data is not None:
            config = json.loads(request.POST.get('config'))
        else:
            config = {}

        #im = Image.open(request.FILES['image']).convert('RGB')
        #im.show()

        if request.POST['isMobile'] == 'true':
            resp = makeGenerationResponseMobile(*generateArticle(input_prompt = request.POST['inputString'],
                                                                 image = request.FILES['image'],
                                                                 config = config))
        else:
            resp = makeGenerationResponse(*generateArticle(input_prompt=request.POST['inputString'],
                                                           image = request.FILES['image'],
                                                           config = config))
        return JsonResponse(resp)
    else:
        print('not post')
