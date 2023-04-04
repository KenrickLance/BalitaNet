<template>
    <header class="header">
        <h1 class="hero-text">AI that Generates Fictional Filipino News Articles</h1>
        <img class="banner-image" src="@/assets/Banner.png" width="750"/>
    </header>
</template>

<style>
.header {
  width: 100%;
  background: #dfdfdf;
  padding-block: 40px;
  padding-inline: 10px;

  border-bottom: 3px solid rgb(63, 63, 63)
}

.hero-text {
  font-family: 'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif;
  letter-spacing: 1px;
  word-spacing: 5px;
  font-weight: bold;
  font-size: 36px;
  color: rgb(0, 0, 0);
  text-align: center;
  margin-bottom: 50px;
}

.banner-image {
  max-width: 100%;
  margin: auto;

}

@media (max-width: 640px) {
  .header {
    padding-block: 20px;
  }
  .hero-text {
    font-size: 22px;
    margin-bottom: 25px;
    letter-spacing: 0px;
    word-spacing: 0px;
    
  }
}
</style>