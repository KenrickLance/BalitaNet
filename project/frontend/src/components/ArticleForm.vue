<template>
  <div class="container all">
    <form @submit.prevent="generateArticle">


      <div class="pick-sample-text">
        Step 1. Choose a news category and upload an image
      </div>

      <div class="my-3">
        <div class="upload-your-own">
          <!--Category Panel-->
          <div class="me-3 category-panel">
            <div class="mb-3 category-title-text">
              Categories:
            </div>
            <div class="category-options">
              <div class="form-check category-option">
                <input v-model="category" class="form-check-input" type="radio" name="category" id="news" value="0" checked>
                <label class="form-check-label category-labels" for="news">
                  <img src="@/assets/icons/general_news.svg" width="30"/>
                  <div>
                    General News
                  </div>
                </label>
              </div>
              <div class="form-check category-option">
                <input v-model="category" class="form-check-input" type="radio" name="category" id="sports" value="1">
                <label class="form-check-label category-labels" for="sports">
                  <img src="@/assets/icons/sports.svg" width="30"/>
                  <div>
                    Sports
                  </div>
                </label>
              </div>
              <div class="form-check category-option">
                <input v-model="category" class="form-check-input" type="radio" name="category" id="entertainment" value="2">
                <label class="form-check-label category-labels" for="entertainment">
                  <img src="@/assets/icons/entertainment.svg" width="30"/>
                  <div>
                    Entertainment
                  </div>
                </label>
              </div>
              <div class="form-check category-option">
                <input v-model="category" class="form-check-input" type="radio" name="category" id="crime" value="3">
                <label class="form-check-label category-labels" for="crime">
                  <img src="@/assets/icons/crime.svg" width="30"/>
                  <div>
                    Crime
                  </div>
                </label>
              </div>
            </div>
          </div>
          <!--Upload file panel-->
          <div class="upload-file-panel">
            
            <div class="upload-form upload-form-web m-3">
              <input name="file" id="entry_value" ref="fileInput" type="file"  @change="onFileUpload">
              <div>
                  <img src="@/assets/upload.png" alt="upload" width="7%" class="upload-icon mx-2" style="display:inline">
                      Upload your image here or
                  <button class="btn bg-color-dblue btn-primary mx-2 px-4 py-3">Browse</button>
              </div>
            </div>

            <div class="upload-form upload-form-mobile m-1">
              <input name="file" id="entry_value" ref="fileInput" type="file"  @change="onFileUpload">
              <div>
                  <img src="@/assets/upload.png" alt="upload" width="7%" class="upload-icon mx-2" style="display:inline">
                  <button class="upload-button btn bg-color-dblue btn-primary">Upload</button>
              </div>
            </div>

            <div class="my-3 container d-flex align-items-center justify-content-center">
              <img class="selected-image"
                   :src="determineSelectedImage">
            </div>
          </div>
        </div>
      </div>

      <div class="or-text my-3">
        OR
      </div>

      <div class="pick-own-text my-3">
        Use a sample image below
      </div>

      <div class="card my-3">
        <div class="card-header">
          <!-- Nav tabs -->
          <ul class="nav nav-tabs card-header-tabs">
            <li class="nav-item sample-tabs">
              <a class="nav-link active" data-bs-toggle="tab" href="#general-tab">General News</a>
            </li>
            <li class="nav-item sample-tabs">
              <a class="nav-link" data-bs-toggle="tab" href="#sports-tab">Sports</a>
            </li>
            <li class="nav-item sample-tabs">
              <a class="nav-link" data-bs-toggle="tab" href="#entertainment-tab">Entertainment</a>
            </li>
            <li class="nav-item sample-tabs">
              <a class="nav-link" data-bs-toggle="tab" href="#crime-tab">Crime</a>
            </li>
          </ul>
        </div>
        <div class="card-body">
        <!-- Tab panes -->
          <div class="tab-content">
            <div v-for="(imageCategory, index) of ['general', 'sports', 'entertainment', 'crime']"
                  :key="index"
                  class="tab-pane container"
                  :class="{'active': imageCategory === 'general'}"
                  :id="`${imageCategory}-tab`">
              <div class="image-pane">
                <label v-for="(imageName, index) of sampleImages[imageCategory]"
                      :key="index"
                      class="image-img">
                  <input v-model="image" @click="onPickImage(imageCategory, imageName)" class="image-radio" type="radio" name='image' :id="imageName" :value="imageName"/>
                  <img class="rounded" :src="require(`@/assets/sampleImages/${imageCategory}/${imageName}`)">
                </label>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="add-title-body-text mt-4">
        Step 2. (Optional) Add a custom title or body to start off your article
      </div>

      <div class="my-3 align-items-center">
        <div>
          <input v-model="title" type="text" class="form-control" id="title" placeholder="Article title">
        </div>
      </div>

      <div class="mb-3">
        <textarea v-model="body" class="form-control" id="body" rows="3" placeholder="Article Body. Make sure you have a title first if you're including a body!"></textarea>
      </div>

      <div v-if="!$store.state.disableGenerationSettings" class="container-fluid">
        <div class="mb-3">
          <label for="temperature" class="form-label">Generation Temperature (Lower: better quality  | Higher: more unique)</label>
          <div class="row">
            <div class="col-4">
              <input type="range" class="form-range" min="0.3" max="1.5" step="0.01" id="temperature" v-model="temperature"> {{ temperature }}
            </div>
          </div>
        </div>
      </div>
      <button type="submit" class="btn btn-primary generate-btn" :disabled="disableSubmit || !imagePicked">Generate</button>
    </form>
  </div>
</template>

<script>
import { nextTick } from 'vue'

import axios from 'axios'

export default {
  emits: ['generateArticle'],
  data() {
    return {
      image: null,
      file: null,
      defaultSelectedImage: 'no-image.jpg',
      selectedCategory: null,
      currentImage: null,
      category: '0',
      title: null,
      body: null,
      temperature: .8,
      disableSubmit: false,
      sampleImages: {
        'general': ['duterte.jpg', 'covid.jpg', 'vaccine.jpg', 'petrol.jpg', 'lrt.jpg', 'roque.jpg'],
        'sports': ['basketball.jpg', 'volleyball.jpg', 'chess.jpg'],
        'entertainment': ['heart.jpg', 'piolo.jpg'],
        'crime': ['fire.jpg', 'drugs.jpg', 'accident.jpg'],
      },
    }
  },

  methods: {
    async generateArticle() {
      console.log('generate article')
      if (this.disableSubmit === true) {
        return null
      }
      this.disableSubmit = true
      let disableSubmitTimer = setTimeout(() => this.disableSubmit = false, 30000)


      let imageToBeSent

      if (this.file !== null) {
        this.currentImage = this.file
        imageToBeSent = this.file
      } else if (this.image !== null) {
        let response = await fetch(require(`@/assets/sampleImages/${this.selectedCategory}/${this.image}`))
        let data = await response.blob();
        let metadata = {
          type: 'image/jpeg'
        };
        imageToBeSent = new File([data], this.image, metadata);
        this.currentImage = imageToBeSent
      } else {
        throw Error('No file or image picked')
      }

      let data = new FormData()
      data.append('inputString', this.makeInputString())
      data.append('image', imageToBeSent, 'image')
      data.append('config',  JSON.stringify({'temperature': this.temperature ?? 1}))
      data.append('isMobile', this.$store.state.isMobileOrTablet)

      axios.post('api/generate',
          data,
          {
            headers: {
            "Content-Type": "multipart/form-data"
            },
            crossDomain: true,
            xsrfCookieName: 'csrftoken',
            xsrfHeaderName: 'X-CSRFTOKEN'
          },
      ).then(resp => {
        console.log('success')

        this.$store.commit('SET_CURRENT_IMAGE', this.currentImage)

        this.$emit('generateArticle', resp.data)
        clearTimeout(disableSubmitTimer)
        this.disableSubmit = false


      nextTick(() => {
        let yOffset = -50
        let el = document.getElementById('articleDisplay')
        let y = el.getBoundingClientRect().top + window.pageYOffset + yOffset
        window.scrollTo({top: y, behavior: 'smooth'})
      })
        
      }).catch(() => {
        console.log('error')
        this.$store.commit('SET_CURRENT_IMAGE', null)
        clearTimeout(disableSubmitTimer)
        this.disableSubmit = false
      })
    },

    makeInputString(){
      let inputString = '<|BOS|><|category|>'
      if (this.category) {
        inputString += this.category
      } else {
        inputString += '0'
      }
      inputString += '<|title|>'
      if (this.title) {
        inputString += this.title
      }
      if (this.body) {
        inputString += '<|body|>'
        inputString += this.body
      }
      console.log(inputString)
      return inputString
    },

    onFileUpload(event) {
      this.image = null;
      this.file = event.target.files[0];
    },

    onPickImage(imageCategory, imageName) {
      this.category = {'general': 0,
                        'sports': 1,
                        'entertainment': 2,
                        'crime': 3}[imageCategory]
      this.selectedCategory = imageCategory
      this.image = imageName

      this.file = null;
      this.$refs.fileInput.value = null;
    },
  },

  computed: {
    determineSelectedImage() {
      if (this.image === null && this.file === null) {
        return require('@/assets/' + this.defaultSelectedImage)
      } else if (this.image !== null) {
         return require(`@/assets/sampleImages/${this.selectedCategory}/${this.image}`)
      } else {
        return URL.createObjectURL(this.file)
      }
    },

    imagePicked() {
      return this.image !== null || this.file !== null
    }
  },
}
</script>

<style scoped>
.image-radio {
    display:none;
}

.image-radio + img {
  cursor: pointer;
}

.image-radio:checked + img {
  outline: 5px solid #f00;
}

.image-pane {
  padding: 10px;
  display: grid;
  grid-template-columns: repeat(auto-fill, 280px);
  justify-content: space-around;
  gap: 1.5em
}

.image-img {
  width: 280px
}

.or-text {
  text-align: center;
  font-weight: 600;
  font-size: 22px;
}

.pick-sample-text{
  text-align: center;
  font-size: 19px;
  font-weight: 600;
  margin-top: 10px;
}

.pick-own-text {
  text-align: center;
  font-size: 19px;
  font-weight: 600;
}

.add-title-body-text {
  text-align: center;
  font-size: 19px;
  font-weight: 600;
}

.upload-your-own {
  display: flex;
  justify-content: space-evenly;
}

.upload-file-panel{
  flex-grow: 0;
}

.category-panel {
  flex-shrink: 0;
  border-right: 2px solid;
  padding-right: 5em;
}

.category-title-text {
  text-align: left;
  font-size: 1.2em;
  font-weight: 500;
}

.category-options {
  text-align: left;
}

.category-option {
  margin-block: 0.5em;
}

.category-labels {
  display: flex;
}

.category-labels div {
  padding-left: 0.25em;
}

.generate-btn {
  width: 100%;
}

.selected-image {
  max-height: 300px;
}

/**   FILE UPLOAD    **/

.upload-form{
position: relative;
max-width: 25em;
height: 100px;
border: 3px dashed grey;
border-radius: 10px;

}
.upload-form div{
width: 100%;
height: 100%;
text-align: center;
line-height: 75px;
position: absolute;
top: 10%;
z-index: -1;
}
.upload-form input{
position: relative;
margin: 0;
padding: 0;
width: 100%;
height: 100%;
outline: none;
opacity: 0;
}
.upload-form button {
    border-radius: 10px;
    padding: 10px 20px;
    background: #1950A3;
    outline: none;
    border: none;
    color: white;
    font-size: 16px;
}
.upload-form div img {
    position: relative;
    top: 2px;
    width: 8%;
}



@media (max-width: 1024px) {
}

@media (min-width: 640px) {
  .upload-form-mobile {
    display: none
  }
}

@media (max-width: 640px) {
  .image-pane {
    padding: 1px;
    display: grid;
    grid-template-columns: repeat(auto-fill, 130px);
    justify-content: space-around;
    gap: 1em
  }

  .image-img {
    width: 130px
  }

  .category-panel {
    border-right: 2px solid;
    padding-right: 1em;
  }

  .upload-form-web {
    display: none;
  }

  .upload-icon {
    min-width: 40px;
  }
  .sample-tabs {
    font-size: 14px
  }

  .upload-form button {
    border-radius: 8px;
    padding: 8px 8px;
    background: #1950A3;
    outline: none;
    border: none;
    color: white;
    font-size: 15px;
  }
}
@media (max-width: 355px) {
  .upload-button {
    display: none !important;
  }
}

</style>