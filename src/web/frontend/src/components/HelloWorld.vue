<template>
  <header>
    <img src="https://svgur.com/i/udg.svg" class="logo">
    <h1 class="header-title">Cavitation ITMO</h1>
  </header>
  <main>
  <div class="section-main">
    <div class="form-group">
      <input
        @change="onFileSelected"
        type="file"
        name="file"
        id="file"
        class="input-file"
      />
      <label for="file" class="btn btn-tertiary js-labelFile">
        <i class="icon fa fa-download"></i>
        <span class="js-fileName">Загрузить файл</span>
      </label>
    </div>
    <div class="section-main-img-title">
      <h1 class="concentration-title">Your concentration: {{ this.class_name }}</h1>
      <img :src="image" v-if="image" class="bubble-img"/>
    </div>
  </div>
</main>
</template>

<script>
import axios from "axios";
export default {
  data() {
    return {
      image: null,
      selectedFile: null,
      class_name: null,
    };
  },
  methods: {
    onFileSelected(event) {
      this.selectedFile = event.target.files[0];
      const config = {
        headers: {
          "content-type": "multipart/form-data",
        },
      };

      const file = new FormData();
      file.append("file", this.selectedFile);
      axios
        .post("http://127.0.0.1:8000/upload_image_for_concentration", file, config)
        .then((res) => {
          console.log(res.status);
          if (res.status == 200) {
            this.class_name = res.data.class_name
          }
          if (res.status == 204) {
            this.class_name = "Не верный тип файла";
          }
        });
        axios
        .post("http://127.0.0.1:8000/upload_image_for_generating_image_base64", file, config)
        .then((res) => {
          console.log(res.status);
          if (res.status == 200) {
            this.image = 'data:image/jpeg;base64,' + res.data.image;
          }
          if (res.status == 204) {
            this.class_name = "Не верный тип файла";
          }
        });
    },
  },
};
</script>


<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Righteous&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Jura:wght@400;600;700&family=Righteous&display=swap');
header {
  display: flex;
  flex-direction: row;
  color: #000000;
  font-family: 'Righteous';
}
h1.header-title {
  margin-left: 20px;
}
.section-main {
  display: flex;
  flex-direction: row;
  justify-content: space-around;
}
.section-main .btn-tertiary {
  margin-top: 20px;
  color: #555;
  padding: 0;
  line-height: 650px;
  width: 650px;
  height: 650px;
  display: block;
  border: 2px solid #555;
}
.section-main .btn-tertiary:hover,
.section-main.btn-tertiary:focus {
  color: #888;
  border-color: #888;
}
.section-main .input-file {
  width: 0.1px;
  height: 0.1px;
  opacity: 0;
  overflow: hidden;
  position: absolute;
  z-index: -1;
}
.section-main .input-file + .js-labelFile {
  overflow: hidden;
  text-align: center;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 0 10px;
  cursor: pointer;
}
.section-main .input-file + .js-labelFile .icon:before {
  content: "\f093";
}
.section-main .input-file + .js-labelFile.has-file .icon:before {
  content: "\f00c";
  color: #5aac7b;
}
.section-main-img-title {
  text-align: center;
  align-items: center;
}
.bubble-img {
  max-width: 600px;
  max-height: 600px;
}
@media (max-width:1550px) {
  .section-main .btn-tertiary {
  line-height: 450px;
  width: 450px;
  height: 450px;
}
  .bubble-img {
    max-width: 400px;
    max-height: 400px;
  }
}
@media (max-width:1300px) {
  .section-main .btn-tertiary {
  line-height: 400px;
  width: 400px;
  height: 400px;
}
  .bubble-img {
    max-width: 400px;
    max-height: 400px;
  }
}
@media (max-width:1100px) {
  .section-main .btn-tertiary {
    margin-top: 20px;
    line-height: 300px;
    width: 300px;
    height: 300px;
}
  .bubble-img {
    max-width: 250px;
    max-height: 250px;
  }
  .concentration-title {
    font-size: 30px;
  }
}

@media (max-width:650px) {
  .section-main {
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .section-main .btn-tertiary {
    margin-top: 20px;
    margin: 0 auto;
    line-height: 400px;
    width: 400px;
    height: 400px;
}
  .bubble-img {
    max-width: 400px;
    max-height: 400px;
  }
  .concentration-title {
    font-size: 40px;
  }
}
@media (max-width:450px) {
  .section-main {
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .section-main .btn-tertiary {
    margin: 0 auto;
    margin-top: 50px;
    line-height: 250px;
    width: 250px;
    height: 250px;
}
  .bubble-img {
    max-width: 250px;
    max-height: 250px;
  }
  .concentration-title {
    margin-top: 30px;
    font-size: 30px;
  }
}
</style>