import axios from 'axios';
import React, { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [upcomingMatches, setUpcomingMatches] = useState([]);

  useEffect(() => {
    fetch("/api/future-matches")
      .then((response) => response.json())
      .then((data) => setUpcomingMatches(data))
  }, []);

  const handleMatchClick = (match) => {
    const matchId = match.id
    axios.post("/api/predict-match", { matchId })
    .then((response) => {
      console.log(response.data)
    })
  }

  return (
    <div className="container">
      <h1 className="header">Premier League Info</h1>

      <section className="section">
        <h2 className="section-title">Upcoming Matches</h2>
        <table className="matches-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Time</th>
              <th>Team</th>
              <th>Opponent</th>
            </tr>
          </thead>
          <tbody>
            {upcomingMatches.map(({ date, time, team, opponent }, i) => (
              <tr key={i}>
                <td>{date}</td>
                <td>{time}</td>
                <td>{team}</td>
                <td>{opponent}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

export default App;
