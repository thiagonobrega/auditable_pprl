var appRouter = function (app) {
    app.get("/", function(req, res) {
      res.status(200).send("Welcome to our restful API");
    });

    app.get("/getblock", function (req, res) {
      var dbu = require("../util/db.js");
      z = dbu.read("./db", 'blk.txt', 10);
      res.status(200).send(z);
    });


    app.get("/getBlk/:idb", function (req, res) {
      var idb = req.params.idb;

      var dbu = require("../util/db.js");
      z = dbu.readById("./db", 'blks.db', idb-1);
      res.status(200).send(z);
    });

  }
  
  module.exports = appRouter;