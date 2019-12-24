let express = require("express");
let router = express.Router();
router.use(express.static(__dirname + "/static"));

router.get("/", function(req, res) {
  //   res.render("selectModel", {});
  console.log("router /predict");
  res.render("predict-with-visuals.html");
});

module.exports = router;
