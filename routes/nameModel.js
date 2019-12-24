let express = require("express");
let router = express.Router();
const axios = require("axios");

global.modelName = "";
router.get("/", function(req, res) {
  res.render("nameModel", { modelName: modelName });
});

router.post("/", function(req, res) {
  modelName = req.body.modelName;
  console.log(modelName);
  console.log("reaching flask");
  const body = {
    selectedModel: selectedModel,
    projectName: projectName,
    modelName: modelName
  };
  console.log("global variables: ", selectedModel, projectName, modelName);
  axios
    .post("http://localhost:5000/train", body)
    .then(res => {
      console.log("flask response: ", res.data);
    })
    .catch(console.log);
  res.render("train", { modelName: modelName });
});

module.exports = router;
