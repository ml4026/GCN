var express = require('express');
var app = express();
var pdfText = require('pdf-text');
var bodyParser = require('body-parser');
var multer  = require('multer')
var bluebird = require("bluebird");
var qs = require('querystring');
var fs = require('fs');

var Model = require('keras-js');
var mongoose = require('mongoose');
var Schema = mongoose.Schema;
/*
 //setting up the mogodb schema
var paperSchema = new Schema({
  name:{
    type: String,
    required:[true, '']
  }
  result:{
    type: String,
    default: 'Other topic types that are not included.'
  }
});

//var paperModel = mongoose.model('paper', paperSchema);

// connect to mongodb
//mongoose.connect('mongodb://localhost/paper');
//mongoose.Promise = global.Promise;
*/

app.use(express.static(__dirname +'./../../')); //serves the index.html

var dictCount = {};
var output = "";


app.post('/file_upload', function(req , res){
    dictCount = {};
    var storage = multer.diskStorage({
        destination: '/Users/kaiwenshi/Documents/app/src/server/uploads'
    });
    var upload = multer({
        storage: storage
    }).any();

    upload(req, res, async err => {
      try {
        if (err) {
          console.log(err);
          return res.end('Error');
        } else {
          await bluebird.each(req.files, async item => {
            const data = fs.readFileSync(item.path);
            const chunks = await bluebird.fromCallback(cb => pdfText(data, cb));
            chunks.forEach((item, index) => {
              newitem = item.split(" ");
              newitem.forEach((item, index) => {
                item = item.toLowerCase();
                if(item==='rule'){
                  console.log('here');
                }
                if (item.match(/^[a-z0-9]+$/i)) {
                  if (item in dictCount) {
                    dictCount[item].count++;

                    console.log(dictCount[item]);
                  } else {
                    dictCount[item] = { word: item, count: 1};
                  }
                }
              });
            });
          });

          var topFeatures = ['genetic','algorithm', 'neural', 'network','theory', 'case', 'base', 'reinforcement', 'learning', 'rule', 'probabilistic', 'method'];
          Object.keys(dictCount).forEach(function(key) {

                if(key in topFeatures){
                  topFeatures[key] = {word:key, count:dictCount[key].count};
                  output+=dictCount[key].count+" ";
                }
          });

          //console.log(topFeatures);
          res.end('File uploaded');

        }
      } catch (e) {
        res.end('Error')
      }
    });
});

app.get('/something', function(req, res){

    res.send({name:'result', features:output});
});


app.listen(11667); //listens on port 3000 -> http://localhost:3000/
