
function readData(fpath,fname) { 
    var fs = require('fs'),
    path = require('path'),    
    // filePath = path.join(__dirname, 'start.html');
    filePath = path.join(fpath, fname);

    var content;
    // First I want to read the file
    fs.readFile(filePath, {encoding: 'utf-8'}, function read(err, data) {
        if (err) {
            throw err;
        }
        var content = data;
        // return {"content": data};
        // Invoke the next step here however you like
        console.log("==>" + content);   // Put all of the code here (not the best solution)
        // processFile();          // Or put the next step in a function and invoke it
    });
    content = data;

    return {"content": content};
}


var data = readData("../db", 'blk.txt');

console.log(data)


// module.exports = {
//     f1: myFunction1,
//     f2: myFunction2
//   };


// console.log("retorno : " + data);
// function readData(fpath,fname) { 
//     var fs = require('fs'),
//     path = require('path'),    
//     // filePath = path.join(__dirname, 'start.html');
//     filePath = path.join(fpath, fname);
//     var retorno="";
    // fs.readFile(filePath, {encoding: 'utf-8'}, function(err,data){
    //     if (!err) {
    //         return data;
    //         console.log('received data: ' + data);
    //         // response.writeHead(200, {'Content-Type': 'text/html'});
    //         // response.write(data);
    //         // response.end();
    //     } else {
    //         console.log(err);
    //     }
    // });
//     return retorno;
// } 
