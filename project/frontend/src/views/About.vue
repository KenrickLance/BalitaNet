<template>
  <div class="container">
    <h1 class="header-about-ai">About the AI</h1>
    <p>
      This AI is based on the GPT-2 language model with modifications to allow conditional generation for images and classes(news categories).
      The language model was trained on the FilipinoNews NLP Dataset which contains over 300k news articles gathered from Filipino news outlets.
      The training data for this language model only goes up to April 2022 and do not include newer articles and so it has limited ability to generate articles about topics on recent affairs.
    </p>
    <a href="https://storage.cloud.google.com/public-kenricklancebunag/Transformer-based%20Conditional%20Language%20Models%20-%20IEOM%20Submission.pdf"
    target="_blank">
      Check out all the details in the research paper!
    </a>
  </div>
</template>

<script>

</script>

<style>
  .header-about-ai {
    margin-top: 1em
  }
</style>