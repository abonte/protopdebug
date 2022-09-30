var main = '1utOGo9u1ck14CY_8tk-HrDAx1gPOThjy8oL9lxxq9q4'

function cleanMainForm() {
  var f = FormApp.openById(main)

  var items = f.getItems();
  for (var e = 0; e < items.length; e++) {
    if (items[e].getType() == FormApp.ItemType.PAGE_BREAK) {
      var pb = items[e].asPageBreakItem()
      if (pb.getTitle().startsWith('Figure')) {
        Logger.log(pb.getTitle())
        f.deleteItem(items[e]);
      }
    }
    if (items[e].getType() == FormApp.ItemType.MULTIPLE_CHOICE) {
        if (items[e].getTitle() === 'The highlighted overlay covers') {
          Logger.log(items[e].getTitle())
          f.deleteItem(items[e]);
        }
    }
    if (items[e].getType() == FormApp.ItemType.IMAGE) {
      Logger.log(items[e].getTitle())
      f.deleteItem(items[e]);
    }
  }
}
function _question(form, img, qid) {
  form.addPageBreakItem().setTitle("Figure " + qid);

  form.addImageItem()
      .setImage(img);

  var item = form.addMultipleChoiceItem();
  item.setTitle('The highlighted overlay covers')
      .setRequired(true)
      .setChoices([
            item.createChoice('some part of the bird'),
            item.createChoice('exclusively (or very nearly so) the background')
       ])
}

function addToMainForm() {
  var form = FormApp.openById(main);

  var i = 0;
  var items = form.getItems();
  for (var e = 0; e < items.length; e++) {
    if (items[e].getType() == FormApp.ItemType.PAGE_BREAK) {
      i = i + 1;
    }
    if (i == 3) { // add question after three sections (excluded the title)
      break
    }
  }
  _add_questions(form);
}

function _add_questions(form) {
  var ff = DriveApp.getFoldersByName('random_form_figures')
  var folder = ff.next();
  var files = folder.getFiles();

  // creates an array of file objects
  let imgs = [];
  while (files.hasNext()) {
      imgs.push(files.next());
  }

  imgs = imgs.sort(function(a, b){
    var aName = a.getName().toUpperCase();
    var bName = b.getName().toUpperCase();

    return aName.localeCompare(bName);
  });

  for(let i=0; i < imgs.length; i++) {
    _question(form, imgs[i], i+1);
    Logger.log(i+1 + " " + imgs[i].getName());
  }
}