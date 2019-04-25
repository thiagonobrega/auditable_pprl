
function readData(fpath,fname) { 
    var fs = require('fs'),
    path = require('path'),    
    // filePath = path.join(__dirname, 'start.html');
    filePath = path.join(fpath, fname);
    console.log(filePath);

    var content = fs.readFileSync(filePath, {encoding: 'utf-8'});
    return content.toString().split("\n");
}

function convert2byteArray(bflen, bfdata) {
  const bf = Buffer.alloc(bflen);
  bfdata.forEach(function(value){
    bf[parseInt(value)]=0xff;
  });

  // return Uint8Array.from(bf)
  return bf;
}

function readBlock(fpath,fname,bflen) {
  var filters = [];
  lines = readData(fpath,fname)
  lines.forEach(function(line){
    var ld = line.replace(/\r$/,'').toString().split(";");
    var eid = ld[0];
    var bf = convert2byteArray(bflen,ld[1].toString().split(","));
    filters.push({
                  id: eid,
                  filter: bf.toJSON().data
    }); // end of push
  }); //end of for each
  return filters;
}

// z = readBlock("./db", 'blk.txt',10);
// console.log(z);

module.exports = {
    // f1: myFunction1,
    read: readBlock
  };

  // z = readBlock("./db", 'blk.txt',10)
// var data = readData("../db", 'blk.txt');
// console.log(data)