//
//  DataGenerator.cpp
//  datagenerator
//
//  Created by Kaiwen Shi on 2018/3/14.
//  Copyright © 2018 Kaiwen Shi. All rights reserved.
//

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <stdlib.h>
#include <chrono>
#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <set>
#include <fstream>
using namespace std;

int randomNumberGenerator(const int lowerBound, const int upperBound){
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dis(lowerBound, upperBound);
    return dis(gen);
}


int main(int argc, const char * argv[]) {
    
    int datasize = 600;
    
    vector<int> category1;
    vector<int> category2;
    int cat1 = 100000;
    int cat2 = 200000;
    
    unordered_map<int, set<int>> cat1Links;
    unordered_map<int, set<int>> cat2Links;
    
    for(int i = 1; i <= datasize; i++){
        category1.push_back(cat1+i);
        category2.push_back(cat2+i);
    }
    
    for(int i = 0; i < category1.size(); i++){
        int countEdges = randomNumberGenerator(10, 100);
        for(int j = 0; j < countEdges; j++){
            int nextEdge = randomNumberGenerator(0, int(category1.size())-1);
            if(nextEdge == i)
                continue;
            cat1Links[category1[i]].insert(category1[nextEdge]);
        }
    }
    
    for(int i = 0; i < category2.size(); i++){
        int countEdges = randomNumberGenerator(10, 100);
        for(int j = 0; j < countEdges; j++){
            int nextEdge = randomNumberGenerator(0, int(category2.size())-1);
            if(nextEdge == i)
                continue;
            cat2Links[category2[i]].insert(category2[nextEdge]);
        }
    }
    
    int numFeatures = 30;
    int noiseLevel = 5;
    float percentile = 0.8;
    
    
    unordered_map<int, vector<int>> features;
    
    
    for(int i = 0; i < category1.size(); i++){
        int current = category1[i];
        if(features.find(current)==features.end()){
            vector<int> newFeature = vector<int>(2*numFeatures, 0);
            
            int counter = 0;
            for(int j = 0; j < numFeatures; j++){
                int value = randomNumberGenerator(0, 1);
                newFeature[j] = value;
                counter+=value;
                if(counter>int(numFeatures*percentile)){
                    break;
                }
            }
            
            for(int i = 0; i < noiseLevel; i++){
                int index = randomNumberGenerator(numFeatures, 2*numFeatures-1);
                newFeature[index] = randomNumberGenerator(0, 1);
            }
            
            features[current] = newFeature;
        }
        
    }
    
    for(int i = 0; i < category2.size(); i++){
        int current = category2[i];
        if(features.find(current)==features.end()){
            vector<int> newFeature = vector<int>(2*numFeatures, 0);
            
            int counter = 0;
            for(int j = numFeatures; j < 2*numFeatures; j++){
                int value = randomNumberGenerator(0, 1);
                newFeature[j] = value;
                counter+=value;
                if(counter>int(numFeatures*percentile)){
                    break;
                }
            }
            
            for(int i = 0; i < noiseLevel; i++){
                int index = randomNumberGenerator(0, numFeatures-1);
                newFeature[index] = randomNumberGenerator(0, 1);
            }
            
            features[current] = newFeature;
        }
    }
    
    ofstream linkFile;
    linkFile.open("data.cites");
    
    for(auto& element:cat1Links){
        int key = element.first;
        set<int> value = element.second;
        for(auto& cited:value){
            linkFile<<key<<"\t"<<cited<<endl;
        }
    }
    
    for(auto& element:cat2Links){
        int key = element.first;
        set<int> value = element.second;
        for(auto& cited:value){
            linkFile<<key<<"\t"<<cited<<endl;
        }
    }
    
    linkFile.close();
    
    ofstream featureFile;
    featureFile.open("data.content");
    
    for(auto& element:features){
        int key = element.first;
        vector<int> feature = element.second;
        if(key>=200000){
            featureFile<<key<<"\t";
            
            for(int i = 0; i < feature.size(); i++){
                featureFile<<feature[i]<<"\t";
            }
            featureFile<<"CAT2"<<endl;
        }else{
            featureFile<<key<<"\t";
            
            for(int i = 0; i < feature.size(); i++){
                featureFile<<feature[i]<<"\t";
            }
            featureFile<<"CAT1"<<endl;
        }
    }
    

    return 0;
}
