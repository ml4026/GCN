import createReactClass from "create-react-class";
import React from 'react'
import ReactDOM from "react-dom";
import DropToUpload from 'react-drop-to-upload';
import Jumbotron from 'reactstrap';
import Container from 'reactstrap';
//import pdfText from 'pdf-text';
import './style/index.css';


var Papers = createReactClass({
            getInitialState: function(){
                        return({
                            papers: []
                        });
                    },
            render: function(){
                var papers = this.state.papers;
                papers = papers.map(function(paper, index){
                    return(
                            <li key={index}>
                              <span className="name">{paper}</span>
                            </li>
                    );
                });
            return(
              <div id="app-wrapper">
                <div className="topnav">
                   <a class="active" href="#home">Homepage</a>
                   <a href="#news">Background</a>
                   <a href="#contact">Contact</a>
                   <a href="#about">About</a>
               </div>
               <div id="homepage">
                <h1 class="title">Online Topic Classifier</h1>
                <h1>Grab a paper and get your classification result!</h1>
                <div id="papers"></div>
                <DropToUpload onDrop={ this.handleDrop } className="dropitem">
                <h1 id="drop-message">Drop a file here to upload</h1>
                </DropToUpload>
                <h1 id="option-message">
                or
                </h1>
                <form id="search" onSubmit={this.handleSubmit}>
                    <label>Please enter the paper url :</label>
                    <input type="text" ref="title" placeholder="Your url" required />
                    <input type="submit" value="Get result" />
                </form>
                <ul>{papers}</ul>

            </div>
          </div>
            );
           },
            handleSubmit: function(e){
                e.preventDefault();
                var title = this.refs.title.value;
                fetch('/api/paper').then(function(data){
                    return data.json();
                }).then( json => {
                    this.setState({
                        papers: ['abcd']
                    });
                    console.log(json);
                });
            },

            handleResult: function(){
              fetch('/something').then(function(data){
                    return data.json();
                }).then( json => {
                    //alert(json['name']);
                    //setTimeout(function(json){

                      var array = json['features'].split(" ");

                      array.forEach((item, index) => {
                        item = parseInt(item);
                      }).then(function(data){
                        const model = new Model({
                          filepath:'/Users/kaiwenshi/downloads/learn-webpack-dev/src/model.bin',
                        });

                        model.ready().then(() => model.predict({
                          input: data,
                        }))
                        .then(({ output }) => {
                          var predictionProbability = -1;
                          var predictedTopic = null;
                          Object.entries(output).forEach(([item, probability]) => {
                            if (probability > predictionProbability) {
                              predictionProbability = probability;
                              predictedTopic = item;
                            }
                          });
                          this.setState({
                           papers: [predictedTopic]
                          });
                        })
                        .catch((error) => {
                          console.log(error);
                        });
                      });
                  //}, 600).bind(this);

                });
            },

            handleDrop(files) {
              //console.log('here');

              var data = new FormData();

              /*pdfText(files[0], function(err, chunks) {
                  console.log(chunks[0]);
              });*/


              //alert((files[0]) instanceof File);
              alert("File uploaded!");
              files.forEach((file, index) => {
                data.append('file' + index, file);
              });

              fetch('/file_upload', {
                method: 'POST',
                body: data
              }).then(this.handleResult);
/*
              fetch('/file_upload').then(function(data){
                  return data.json();
              }).then( json => {
                  //this.setState({
                  //    papers: json
                  //});
                  console.log(json);
              });*/
            }
});
ReactDOM.render(<Papers/>, document.getElementById('app'));
