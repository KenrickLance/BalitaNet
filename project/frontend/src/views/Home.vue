<template>
  <Banner class="banner"/>
  <ArticleForm @generateArticle="(x) => generatedArticle = x"/>
  <hr class="container-lg" id="articleDisplay">
  <div>
  <ArticleDisplay v-if="generatedArticle"
                  v-bind="generatedArticle"/>
  </div>
</template>

<script>
  import ArticleForm from "@/components/ArticleForm";
  import ArticleDisplay from "@/components/ArticleDisplay";
  import Banner from "@/components/Banner";

  export default {
    components: {
      ArticleForm,
      ArticleDisplay,
      Banner
    },

    data() {
      return {
        generatedArticle: null,
      }
    },
  }
</script>

<style scoped>
.banner {
  margin-bottom: 30px;
}
</style>