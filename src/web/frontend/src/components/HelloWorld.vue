<template>
  <link
    rel="stylesheet"
    href="https://maxcdn.bootstrapcdn.com/font-awesome/4.5.0/css/font-awesome.min.css"
  />

  <div class="example-2">
    <div class="form-group">
      <input
        @change="onFileSelected"
        type="file"
        name="file"
        id="file"
        class="input-file"
      />
      <label for="file" class="btn btn-tertiary js-labelFile">
        <i class="icon fa fa-check"></i>
        <span class="js-fileName">Загрузить файл</span>
      </label>
    </div>
    <h1>{{ this.class_name }}</h1>
  </div>
</template>

<script>
import axios from "axios";
export default {
  data() {
    return {
      selectedFile: null,
      class_name: null,
    };
  },
  methods: {
    onFileSelected(event) {
      this.selectedFile = event.target.files[0];
      // console.log(event)
      // var user = {
      //   user: "Fred"
      // }
      const config = {
        headers: {
          "content-type": "multipart/form-data",
        },
      };
      const file = new FormData();
      file.append("file", this.selectedFile);
      axios
        .post("http://127.0.0.1:8000/upload_file", file, config)
        .then((res) => {
          console.log(res.status);
          if (res.status == 200) {
            this.class_name = res.data.class_name
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
.example-2 .btn-tertiary {
  color: #555;
  padding: 0;
  line-height: 400px;
  width: 800px;
  margin: auto;
  display: block;
  border: 2px solid #555;
}
.example-2 .btn-tertiary:hover,
.example-2 .btn-tertiary:focus {
  color: #888;
  border-color: #888;
}
.example-2 .input-file {
  width: 0.1px;
  height: 0.1px;
  opacity: 0;
  overflow: hidden;
  position: absolute;
  z-index: -1;
}
.example-2 .input-file + .js-labelFile {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 0 10px;
  cursor: pointer;
}
.example-2 .input-file + .js-labelFile .icon:before {
  content: "\f093";
}
.example-2 .input-file + .js-labelFile.has-file .icon:before {
  content: "\f00c";
  color: #5aac7b;
}
</style>
