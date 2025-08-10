import React from "react";
import "./App.css";

function App() {
  const teams = [
    "Arsenal",
    "Manchester City",
    "Manchester United",
    "Chelsea",
    "Liverpool",
    "Tottenham",
    "Leicester City",
    "Everton",
    "Aston Villa",
  ];

  const upcomingMatches = [
    { date: "2025-08-15", team: "Arsenal", opponent: "Chelsea" },
    { date: "2025-08-16", team: "Manchester City", opponent: "Liverpool" },
    { date: "2025-08-17", team: "Manchester United", opponent: "Tottenham" },
  ];

  return (
    <div className="container">
      <h1 className="header">Premier League Info</h1>

      <section className="section">
        <h2 className="section-title">Top Teams</h2>
        <ul className="team-list">
          {teams.map((team, i) => (
            <li key={i}>{team}</li>
          ))}
        </ul>
      </section>

      <section className="section">
        <h2 className="section-title">Upcoming Matches</h2>
        <table className="matches-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Team</th>
              <th>Opponent</th>
            </tr>
          </thead>
          <tbody>
            {upcomingMatches.map(({ date, team, opponent }, i) => (
              <tr key={i}>
                <td>{date}</td>
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
