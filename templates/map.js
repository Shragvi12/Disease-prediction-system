function initMap() {
    const userLocation = { lat: 12.9716, lng: 77.5946 }; // Example coordinates
    const map = new google.maps.Map(document.getElementById("map"), {
      center: userLocation,
      zoom: 14,
    });
  
    new google.maps.Marker({
      position: userLocation,
      map: map,
      title: "You are here",
    });
  
    const request = {
      location: userLocation,
      radius: '50000',
      type: ['hospital'],
    };
  
    const service = new google.maps.places.PlacesService(map);
    service.nearbySearch(request, (results, status) => {
      if (status === google.maps.places.PlacesServiceStatus.OK) {
        for (let i = 0; i < results.length; i++) {
          new google.maps.Marker({
            position: results[i].geometry.location,
            map: map,
            title: results[i].name,
          });
        }
      }
    });
  }
  