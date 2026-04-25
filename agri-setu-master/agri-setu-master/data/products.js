export const getProduct = (productId) => {
  let matchingProduct;


  products.forEach((product) => {
    if (product.id === productId) {
      matchingProduct = product;
    }

  });
  return matchingProduct;


}



export const products = [
  { "id": '1', "name": "Paddy", "image": "images/products/paddy.jpg", "price": 4500, "category": "Grains & Pulses", "keywords": ["paddy", "rice grain", "kerala", "farming"] },
  { "id": '2', "name": "Rice", "image": "images/products/rice.jpg", "price": 6000, "category": "Grains & Pulses", "keywords": ["rice", "white rice", "grain", "staple"] },
  { "id": '3', "name": "Pineapple", "image": "images/products/pineapple.jpg", "price": 3500, "category": "Vegetables & Fruits", "keywords": ["pineapple", "fruit", "tropical"] },
  { "id": '4', "name": "Arecanut", "image": "images/products/arecanut.jpg", "price": 8200, "category": "Spices & Cash Crops", "keywords": ["arecanut", "betel nut", "plantation"] },
  { "id": '5', "name": "Turmeric", "image": "images/products/turmeric.jpg", "price": 5000, "category": "Spices & Cash Crops", "keywords": ["turmeric", "spice", "medicinal"] },
  { "id": '6', "name": "Cashew", "image": "images/products/cashew.jpg", "price": 9500, "category": "Spices & Cash Crops", "keywords": ["cashew", "nuts", "export"] },
  { "id": '7', "name": "Coconut", "image": "images/products/coconut.jpg", "price": 3000, "category": "Spices & Cash Crops", "keywords": ["coconut", "kerala", "oil"] },
  { "id": '8', "name": "Ginger", "image": "images/products/ginger.jpg", "price": 4800, "category": "Spices & Cash Crops", "keywords": ["ginger", "spice", "fresh"] },
  { "id": '9', "name": "Pepper", "image": "images/products/pepper.jpg", "price": 12000, "category": "Spices & Cash Crops", "keywords": ["pepper", "black pepper", "export"] },
  { "id": '10', "name": "Coffee", "image": "images/products/coffee.jpg", "price": 11000, "category": "Spices & Cash Crops", "keywords": ["coffee", "beans", "arabica"] },
  { "id": '11', "name": "Papaya", "image": "images/products/papaya.jpg", "price": 2800, "category": "Vegetables & Fruits", "keywords": ["papaya", "fruit", "fresh"] },
  { "id": '12', "name": "Jackfruit", "image": "images/products/jackfruit.jpg", "price": 3200, "category": "Vegetables & Fruits", "keywords": ["jackfruit", "kerala", "organic"] },
  { "id": '13', "name": "Banana", "image": "images/products/banana.jpg", "price": 2600, "category": "Vegetables & Fruits", "keywords": ["banana", "nendran", "fruit"] },
  { "id": '14', "name": "Mango", "image": "images/products/mango.jpg", "price": 4000, "category": "Vegetables & Fruits", "keywords": ["mango", "sweet", "tropical"] },
  { "id": '15', "name": "Rubber", "image": "images/products/rubber.jpg", "price": 7000, "category": "Spices & Cash Crops", "keywords": ["rubber", "latex", "plantation"] },
  { "id": '16', "name": "Cardamom", "image": "images/products/cardamom.jpg", "price": 15000, "category": "Spices & Cash Crops", "keywords": ["cardamom", "spice", "export"] },
  { "id": '17', "name": "Tomato", "image": "images/products/tomato.jpg", "price": 2200, "category": "Vegetables & Fruits", "keywords": ["tomato", "vegetable", "fresh"] },
  { "id": '18', "name": "Onion", "image": "images/products/onion.jpg", "price": 2400, "category": "Vegetables & Fruits", "keywords": ["onion", "vegetable", "kitchen"] },
  { "id": '19', "name": "Potato", "image": "images/products/potato.jpg", "price": 2100, "category": "Vegetables & Fruits", "keywords": ["potato", "vegetable", "staple"] },
  { "id": '21', "name": "Cabbage", "image": "images/products/cabbage.jpg", "price": 1800, "category": "Vegetables & Fruits", "keywords": ["cabbage", "vegetable", "fresh"] },
  { "id": '22', "name": "Carrot", "image": "images/products/carrot.jpg", "price": 2000, "category": "Vegetables & Fruits", "keywords": ["carrot", "vegetable", "organic"] },
  { "id": '23', "name": "Beetroot", "image": "images/products/beetroot.jpg", "price": 1900, "category": "Vegetables & Fruits", "keywords": ["beetroot", "vegetable", "healthy"] },
  { "id": '24', "name": "Green Gram", "image": "images/products/greengram.jpg", "price": 5400, "category": "Grains & Pulses", "keywords": ["green gram", "pulse", "protein"] },
  { "id": '25', "name": "Black Gram", "image": "images/products/blackgram.jpg", "price": 5600, "category": "Grains & Pulses", "keywords": ["black gram", "urad dal", "pulse"] }
];

// helper to look up a product by id

